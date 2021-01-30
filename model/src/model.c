///////////////////////////////////////////////////////////////////////////////
// 1. Includes, macros, etc.
///////////////////////////////////////////////////////////////////////////////

// includes
#include <unistd.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_types.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_interp.h>
#include <omp.h>

// macros: discretization
#define NI 30 // number of industries
#define NZ 201 // productivity shock grid size
#define NT 100 // simulation length
#define NU 20 // number of periods with uncertainty
#define NS 100 // number of simulations
#define NF 1000 // simulation population size

// macros: paralellization
#ifdef _OPENMP
#define PAR 1
#else
#define PAR 0
#endif

// macros: tolerances
const int root_max_iter = 100;
const double root_tol_rel = 1.0e-6;
const double root_tol_abs = 1.0e-6;
const double delta_deriv = 1.0e-9;
const double delta_root = 1.0e-9;
const double policy_tol_abs = 1.0e-8;
const int policy_max_iter = 2000;
const double x_grid_ub_mult = 10.0;
const double x_grid_exp = 1.0;

// print verbose output y/n
const int verbose=1;

// initialize all elements of an array to the same numeric value
void set_all_v(double * v, int n, double val)
{
  int i;
  for(i=0; i<n; i++)
    {
      v[i]=val;
    }
}

// macro used for applying above to slices of multidimensional statically allocated arrays
#define SET_ALL_V(v,n,val) ( set_all_v( (double *)(v), (n), (val) ) )

// write array to text file
void write_vector(const double * v, int n, char * fname)
{
  FILE * file = fopen(fname,"wb");
  int i;
  for(i=0; i<n; i++)
    {
      fprintf(file,"%0.16f\n",v[i]);
    }
  fclose(file);
}

// macro used for writing slices of multidimensional statically allocated arrays
#define WRITE_VECTOR(v,n,fname) ( write_vector( (double *)(v), (n), (fname) ) )

// linspace
void linspace(double lo, double hi, int n, double * v)
{
  double d=(hi-lo)/(n-1.0);
  v[0]=lo;
  int i=0;
  for(i=1;i<n;i++)
    {
      v[i] = v[i-1]+d;
    }
}

// linear interpolation
static inline double interp(gsl_interp_accel * acc, const double *xa, const double *ya, int n, double x)
{
  double x0=0.0;
  double x1=0.0;
  double xd=0.0;
  double q0=0.0;
  double q1=0.0;
  double retval=0.0;

  int ix = gsl_interp_accel_find(acc, xa, n, x);

  if(ix==0)
    {
      x0 = xa[0];
      x1 = xa[1];
      xd = x1-x0;
      q0 = ya[0];
      q1 = ya[1];
    }
  else if(ix==n-1)
    {
      x0 = xa[n-2];
      x1 = xa[n-1];
      xd = x1-x0;
      q0 = ya[n-2];
      q1 = ya[n-1];
    }
  else
    {
      x0 = xa[ix];
      x1 = xa[ix+1];
      xd = x1-x0;
      q0 = ya[ix];
      q1 = ya[ix+1];
    }

  retval = ( q0*(x1-x) + q1*(x-x0) ) / xd;
  return retval;
}

// linear interpolation with index already calculated
static inline double interp_with_ix(const double *xa, const double *ya, int n, double x, int ix)
{
  double x0=0.0;
  double x1=0.0;
  double xd=0.0;
  double q0=0.0;
  double q1=0.0;
  double retval=0.0;

  if(ix==0)
    {
      x0 = xa[0];
      x1 = xa[1];
      xd = x1-x0;
      q0 = ya[0];
      q1 = ya[1];
    }
  else if(ix==n-1)
    {
      x0 = xa[n-2];
      x1 = xa[n-1];
      xd = x1-x0;
      q0 = ya[n-2];
      q1 = ya[n-1];
    }
  else
    {
      x0 = xa[ix];
      x1 = xa[ix+1];
      xd = x1-x0;
      q0 = ya[ix];
      q1 = ya[ix+1];
    }

  retval = ( q0*(x1-x) + q1*(x-x0) ) / xd;
  return retval;
}

// stationary distribution of Markov chain
void markov_stationary_dist(int n, double P[n][n], double p[n])
{
  SET_ALL_V(p,n,1.0/n);
  double diff=-HUGE_VAL;
  do
    {
      diff=-HUGE_VAL;
      
      double tmp[n];
      
      for(int i=0; i<n; i++)
	{
	  tmp[i]=0.0;
	  
	  for(int j=0; j<n; j++)
	    {
	      tmp[i] += P[j][i]*p[j];
	    }
	}
      for(int i=0; i<n; i++)
	{
	  if(fabs(tmp[i]-p[i])>diff)
	    {
	      diff=fabs(tmp[i]-p[i]);
	    }
	}
       for(int i=0; i<n; i++)
	 {
	   p[i]=tmp[i];
	 }
    }
  while(diff>1.0e-11);
}

// horizontal line breaks
void linebreak()
{
  printf("\n////////////////////////////////////////////////////////////////////////////\n\n");
}

void linebreak2()
{ 
  printf("\n----------------------------------------------------------------------------\n");
}

///////////////////////////////////////////////////////////////////////////////
// 2. Declarations of parameters, grids, and inline functions
///////////////////////////////////////////////////////////////////////////////

// parameters
double W = 0.0; // wage (note: represents normalization of export country GDP per capita relative to representative destination)
double Q = 0.0; // discount factor
double delta0[NI] = {0.0}; // survival rate parameter 1
double delta1[NI] = {0.0}; // survival rate parameter 2
double delta[NI][NZ] = {{0.0}}; // survival rate vector
double theta = 0.0; // EoS between varieties
double theta_hat = 0.0; // = (1/theta)*(theta/(theta-1))^(1-theta)
double sig_z[NI] = {0.0}; // stochastic productivity dispersion
double rho_z[NI] = {0.0}; // stochastic productivity persistence
double mu_e[NI] = {0.0}; // new entrant productivity
double kappa0[NI] = {0.0}; // entry cost
double kappa1[NI] = {0.0}; // continuation cost
double xi[NI] = {0.0}; // iceberg cost in high state
double rho0[NI] = {0.0}; // transition probability from low state
double rho1[NI] = {0.0}; // transition probability from high state

// productivity shock grid
double z_grid[NI][NZ] = {{0.0}}; // grid
double z_hat[NI][NZ] = {{0.0}}; // z^{theta-1} grid
double z_ucond_probs[NI][NZ] = {{0.0}}; // ergodic probabilities
double z_ucond_cumprobs[NI][NZ] = {{0.0}}; // cumulative ergodic probabilities
double z_trans_probs[NI][NZ][NZ] = {{{0.0}}}; // transition probabilities
double z_trans_cumprobs[NI][NZ][NZ] = {{{0.0}}}; // cumultative transition probabilities

// tariffs
char industry[NI][128] = {{""}};
double tau[NI] = {0.0}; // trade cost before liberaliztion
double tau_hat[NI] = {0.0}; // = tau^(1-theta)
double pi_hat[NI] = {0.0}; // theta_hat*tau_hat
double tau2[NI] = {0.0}; // trade cost after liberalization
double tau_hat2[NI] = {0.0}; // = tau2^(1-theta)
double pi_hat2[NI] = {0.0}; // theta_hat*tau_hat2
double tpu_prob = 0.5; // probability of reform reverting
int made_up_tariffs=1;
  
// discretization of productivity shock process
void discretize_z(int i)
{
  int n = NZ;
  double inprob = 1.0e-8;
  double lo = gsl_cdf_ugaussian_Pinv(inprob)*sig_z[i]*1.5;
  double hi = -gsl_cdf_ugaussian_Pinv(inprob)*sig_z[i]*1.5;
  double ucond_std = sqrt(sig_z[i]*sig_z[i]/(1.0-rho_z[i]*rho_z[i]));
  double d = (hi-lo)/(n-1.0);
  linspace(lo,hi,n,z_grid[i]);
  
  for(int iz=0; iz<n; iz++)
    {
      double x = z_grid[i][iz];

      double sum=0.0;
      for(int izp=0; izp<n; izp++)
	{
	  double y = z_grid[i][izp];
	  
	  z_trans_probs[i][iz][izp] = (gsl_cdf_ugaussian_P( (y + d/2.0 - rho_z[i]*x) / sig_z[i] ) -
				       gsl_cdf_ugaussian_P( (y - d/2.0 - rho_z[i]*x) / sig_z[i] ));
	  sum += z_trans_probs[i][iz][izp];
	}
      for(int izp=0; izp<n; izp++)
	{
	  z_trans_probs[i][iz][izp] = z_trans_probs[i][iz][izp]/sum;
	}
    }

  double sum=0.0;
  for(int iz=0; iz<n; iz++)
    {
      double x = z_grid[i][iz];
      
      z_ucond_probs[i][iz] = (gsl_cdf_ugaussian_P( (x + mu_e[i] + d/2.0) / ucond_std ) -
			  gsl_cdf_ugaussian_P( (x + mu_e[i] - d/2.0) / ucond_std ));
      sum += z_ucond_probs[i][iz];
    }
  for(int iz=0; iz<n; iz++)
    {
      z_ucond_probs[i][iz] = z_ucond_probs[i][iz]/sum;
    }

  sum=0.0;
  for(int iz=0; iz<n; iz++)
    {
      z_grid[i][iz] = exp(z_grid[i][iz]);
      //z_hat[i][iz] = pow(z_grid[i][iz],theta-1.0);
      z_hat[i][iz] = z_grid[i][iz];
      sum += z_ucond_probs[i][iz];
      z_ucond_cumprobs[i][iz] = sum;

      double sum2=0.0;
      for(int izp=0; izp<n; izp++)
	{
	  sum2 += z_trans_probs[i][iz][izp];
	  z_trans_cumprobs[i][iz][izp] = sum2;
	}
    }
}

// survival probability vector
void calc_death_probs(int i)
{
  for(int iz=0; iz<NZ; iz++)
    {
      double death_prob=fmax(0.0,fmin(exp(-delta0[i]*z_hat[i][iz])+delta1[i],1.0));
      delta[i][iz] = 1.0-death_prob;
    }
}

// assigned parameters and initial guesses
int init_params()
{  
  // initial guesses!!!
  W = 1.0;
  Q = 0.85;
  theta = 5.0;
  theta_hat = (1.0/theta) * pow(theta/(theta-1.0),1.0-theta);

  for(int i=0; i<NI; i++)
    {
      delta0[i] = 21.04284098;
      delta1[i] = 0.02258301;
      sig_z[i] = 1.27;
      rho_z[i] =  0.65;
      kappa0[i] = 0.055;
      kappa1[i] = 0.03;
      xi[i]=1.71823402/1.07027152;
      rho0[i]=0.91571120;
      rho1[i]=0.91571120;
    }
  
  // set all destination-specific variables to mean values... we will use the
  // array of destinations in parallelizing the calibration
  if(!made_up_tariffs)
    {
      FILE * file = fopen("input/tariffs.txt","r");
      if(!file)
	{
	  printf("Failed to open file with tariff data!\n");
	  return 1;
	}
      else
	{
	  char buffer[128];
	  double tau_, tau2_;
	  int got;
	  
	  for(int i=0; i<NI; i++)
	    {
	      got = fscanf(file,"%s %lf %lf",buffer,&tau_,&tau2_);
	      if(got!=2)
		{
		  printf("Failed to load data for industry %d!\n",i);
		  fclose(file);
		  return 1;
		}
	      else
		{
		  tau[i] = tau_;
		  tau2[i] = tau2_;
		  tau_hat[i] = pow(tau[i],1.0-theta);
		  tau_hat2[i] = pow(tau2[i],1.0-theta);
		  strncpy(industry[i],buffer,128);
		  pi_hat[i] = theta_hat * tau_hat[i];
		  pi_hat2[i] = theta_hat * tau_hat2[i];
		}
	    }

	  return 0;
	}
    }
  else
    {
      for(int i=0; i<NI; i++)
	{
	  tau[i] = 1.0+((double)(i+1))/((double)NI);
	  //tau[i] = 1.3;
	  tau2[i] = 1.0;
	  tau_hat[i] = pow(tau[i],1.0-theta);
	  tau_hat2[i] = pow(tau2[i],1.0-theta);
	  sprintf(industry[i],"%d",i);
	  pi_hat[i] = theta_hat * tau_hat[i];
	  pi_hat2[i] = theta_hat * tau_hat2[i];
	}
      return 0;
    }
}


///////////////////////////////////////////////////////////////////////////////
// 3. Dynamic program
///////////////////////////////////////////////////////////////////////////////

double V[NI][NZ][3] = {{{0.0}}}; // pre-liberalization value function
double V2[NI][NZ][3] = {{{0.0}}}; // post-liberalization value function
double V3[NI][NZ][3] = {{{0.0}}}; // TPU value function

double EV[NI][NZ][3] = {{{0.0}}}; // pre-liberalization continuation value
double EV2[NI][NZ][3] = {{{0.0}}}; // post-liberalization continuation value
double EV3[NI][NZ][3] = {{{0.0}}}; // continuation value with TPU

int gex[NI][NZ][3] = {{{0}}}; // pre-liberalization policy function
int gex2[NI][NZ][3] = {{{0}}}; // post-liberalization policy function
int gex3[NI][NZ][3] = {{{0}}}; // policy-function with permanent TPU
int gex4[NI][NZ][3][NU] = {{{{0}}}}; // policy-function with temporary TPU
int policy_solved_flag[NI] = {0};

// initial guess for value functions
void init_dp_objs()
{
  for(int i=0; i<NI; i++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  V[i][iz][0] = 0.0;
	  V[i][iz][1] = pi_hat[i]*pow(xi[i],1.0-theta)*z_hat[i][iz]/Q;
	  V[i][iz][2] = pi_hat[i]*z_hat[i][iz]/Q;
	  
	  V2[i][iz][0] = 0.0;
	  V2[i][iz][1] = pi_hat2[i]*pow(xi[i],1.0-theta)*z_hat[i][iz]/Q;
	  V2[i][iz][2] = pi_hat2[i]*z_hat[i][iz]/Q;
	}
    }
}

// calculate continuation values
void calc_EV(int i)
{
  for(int z=0; z<NZ; z++)
    {
      for(int e=0; e<3; e++)
	{
	  EV[i][z][e]=0.0;
	  EV2[i][z][e]=0.0;
	  EV3[i][z][e]=0.0;
	  for(int zp=0; zp<NZ; zp++)
	    {
	      if(z_trans_probs[i][z][zp]>1.0e-11)
		{
		  EV[i][z][e] += Q*delta[i][z]*V[i][zp][e]*z_trans_probs[i][z][zp];
		  EV2[i][z][e] += Q*delta[i][z]*V2[i][zp][e]*z_trans_probs[i][z][zp];
		  //EV3[i][z][e] = tpu_prob * EV[i][z][e] + (1.0-tpu_prob)*EV2[i][z][e];
		  EV3[i][z][e] += Q*delta[i][z]*z_trans_probs[i][z][zp] * (tpu_prob*V[i][zp][e] + (1.0-tpu_prob)*V3[i][zp][e]);
		}
	    }
	} 
    }
  
}

// Bellmabn equation iteration
void iterate_policies(int i, double * maxdiff, int imaxdiff[3])
{
  *maxdiff=-HUGE_VAL;

  calc_EV(i);
  
  for(int z=0; z<NZ; z++)
    {
      // first compute policy functions -------------------
      
      // pre-liberalzation
      double pi = pi_hat[i]*z_hat[i][z];
      
      if(EV[i][z][0]<
	 pi*pow(xi[i],1.0-theta) + EV[i][z][1] - kappa0[i])
	{
	  gex[i][z][0] = 1;
	}
      else
	{
	  gex[i][z][0] = 0;
	}

      if(EV[i][z][0]<
	 pi*pow(xi[i],1.0-theta) - kappa1[i] +
	 rho0[i]*EV[i][z][1] + (1.0-rho0[i])*EV[i][z][2])
	{
	  gex[i][z][1] = 1;
	}
      else
	{
	  gex[i][z][1] = 0;
	}
	  
      if(EV[i][z][0]<
	 pi - kappa1[i] +
	 (1.0-rho1[i])*EV[i][z][1] + rho1[i]*EV[i][z][2])
	{
	  gex[i][z][2] = 1;
	}
      else
	{
	  gex[i][z][2] = 0;
	}

      // post-liberalzation
      pi = pi_hat2[i]*z_hat[i][z];
      if(EV2[i][z][0]<
	 pi*pow(xi[i],1.0-theta) + EV2[i][z][1] - kappa0[i])
	{
	  gex2[i][z][0] = 1;
	}
      else
	{
	  gex2[i][z][0] = 0;
	}

      if(EV2[i][z][0]<
	 pi*pow(xi[i],1.0-theta) - kappa1[i] +
	 rho0[i]*EV2[i][z][1] + (1.0-rho0[i])*EV2[i][z][2])
	{
	  gex2[i][z][1] = 1;
	}
      else
	{
	  gex2[i][z][1] = 0;
	}
	  
      if(EV2[i][z][0]<
	 pi - kappa1[i] +
	 (1.0-rho1[i])*EV2[i][z][1] + rho1[i]*EV2[i][z][2])
	{
	  gex2[i][z][2] = 1;
	}
      else
	{
	  gex2[i][z][2] = 0;
	}
      
      // update continuation values and check convergence ---------------
      // pre-reform
      pi = pi_hat[i]*z_hat[i][z];
      double tmp0 = fmax(EV[i][z][0],
			 pi*pow(xi[i],1.0-theta) - kappa0[i] +
			 EV[i][z][1]);
	  
      double tmp1 = fmax(EV[i][z][0],
			 pi*pow(xi[i],1.0-theta) - kappa1[i] +
			 rho0[i]*EV[i][z][1] +
			 (1.0-rho0[i])*EV[i][z][2]);

      double tmp2 = fmax(EV[i][z][0],
			 pi - kappa1[i] +
			 (1.0-rho1[i])*EV[i][z][1] +
			 rho1[i]*EV[i][z][2]);

	  
      double diff0 = fabs(tmp0-V[i][z][0]);
      double diff1 = fabs(tmp1-V[i][z][1]);
      double diff2 = fabs(tmp2-V[i][z][2]);

      if(diff0>*maxdiff)
	{
	  *maxdiff=diff0;
	  imaxdiff[0]=z;
	  imaxdiff[1]=0;
	  imaxdiff[2] = 0;
	}

      if(diff1>*maxdiff)
	{
	  *maxdiff=diff1;
	  imaxdiff[0]=z;
	  imaxdiff[1]=1;
	  imaxdiff[2] = 0;
	}
	  
      if(diff2>*maxdiff)
	{
	  *maxdiff=diff2;
	  imaxdiff[0]=z;
	  imaxdiff[1]=2;
	  imaxdiff[2] = 0;
	}

      V[i][z][0] = tmp0;
      V[i][z][1] = tmp1;
      V[i][z][2] = tmp2;

      // post-reform
      pi = pi_hat2[i]*z_hat[i][z];
      tmp0 = fmax(EV2[i][z][0],
			 pi*pow(xi[i],1.0-theta) - kappa0[i] +
			 EV2[i][z][1]);
	  
      tmp1 = fmax(EV2[i][z][0],
		  pi*pow(xi[i],1.0-theta) - kappa1[i] +
			 rho0[i]*EV2[i][z][1] +
			 (1.0-rho0[i])*EV2[i][z][2]);

      tmp2 = fmax(EV2[i][z][0],
			 pi - kappa1[i] +
			 (1.0-rho1[i])*EV2[i][z][1] +
			 rho1[i]*EV2[i][z][2]);

	  
      diff0 = fabs(tmp0-V2[i][z][0]);
      diff1 = fabs(tmp1-V2[i][z][1]);
      diff2 = fabs(tmp2-V2[i][z][2]);

      if(diff0>*maxdiff)
	{
	  *maxdiff=diff0;
	  imaxdiff[0]=z;
	  imaxdiff[1]=0;
	  imaxdiff[2]=1;
	}

      if(diff1>*maxdiff)
	{
	  *maxdiff=diff1;
	  imaxdiff[0]=z;
	  imaxdiff[1]=1;
	  imaxdiff[2]=1;
	}
	  
      if(diff2>*maxdiff)
	{
	  *maxdiff=diff2;
	  imaxdiff[0]=z;
	  imaxdiff[1]=2;
	  imaxdiff[2]=1;
	}

      V2[i][z][0] = tmp0;
      V2[i][z][1] = tmp1;
      V2[i][z][2] = tmp2;
    }
}

// solve policy function for industry i
int solve_policies(int i)
{
  time_t start, stop;
  time(&start);

  init_dp_objs(i);

  int status = 0;
  double maxdiff = 999;
  int imaxdiff[3];
  
  int iter=0;
  do
    {
      iter++;
      iterate_policies(i,&maxdiff,imaxdiff);

      if(verbose==3)
	printf("\t\tIter %d, diff = %0.2g\n",iter,maxdiff);
    }
  while(maxdiff>policy_tol_abs && iter < policy_max_iter);

  time(&stop);

  if(iter==policy_max_iter)
    {
      status=1;
      if(verbose==2)
	printf("\tValue function iteration failed for industry %d! Diff = %0.4g\n",
	       i,maxdiff);
    }
  else
    {
      if(verbose==2)
	{
	  printf("\tValue function for industry %d converged to tolerance %0.4g in %d iterations/%0.0f seconds!\n",
		 i,maxdiff,iter,difftime(stop,start));
	}
    }

  return status;

}

// solve policies for all industries in parallel
int solve_policies2()
{
  if(verbose)
    printf("\nSolving dynamic program...\n");

  time_t start, stop;
  time(&start);

  int cnt=0;
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<NI; i++)
    {
      policy_solved_flag[i] = solve_policies(i);
      cnt += policy_solved_flag[i];
    }

  time(&stop);
  
  if(verbose)
    {
      printf("Finished dynamic programs in %0.0f seconds. %d failed to converge.\n",difftime(stop,start),cnt);
    }
  
  return 0;  
}

// Bellman equation iteration for temporary TPU transition
void iterate_policies_tpu(int i, int t, double * maxdiff, int imaxdiff[3])
{
  *maxdiff=-HUGE_VAL;
    
  calc_EV(i);
  
  for(int z=0; z<NZ; z++)
    {
      // first compute policy functions -------------------
      
      double pi = pi_hat2[i]*z_hat[i][z];     	  
      if(EV3[i][z][0]<
	 pi*pow(xi[i],1.0-theta) + EV3[i][z][1] - kappa0[i])
	{
	  if(t>=0)
	    {
	      gex4[i][z][0][t] = 1;
	    }

	  gex3[i][z][0] = 1;
	}
      else
	{
	  if(t>=0)
	    {
	      gex4[i][z][0][t] = 0;
	    }
	  
	  gex3[i][z][0] = 0;
	}

      if(EV3[i][z][0]<
	 pi*pow(xi[i],1.0-theta) - kappa1[i] +
	 rho0[i]*EV3[i][z][1] + (1.0-rho0[i])*EV3[i][z][2])
	{
	  if(t>=0)
	    {
	      gex4[i][z][1][t] = 1;
	    }
	  
	  gex3[i][z][1] = 1;
	}
      else
	{
	  if(t>=0)
	    {
	      gex4[i][z][1][t] = 0;
	    }
	  
	  gex3[i][z][1] = 0;
	}
	  
      if(EV3[i][z][0]<
	 pi - kappa1[i] +
	 (1.0-rho1[i])*EV3[i][z][1] + rho1[i]*EV3[i][z][2])
	{
	  if(t>=0)
	    {
	      gex4[i][z][2][t] = 1;
	    }

	  gex3[i][z][2] = 1;
	}
      else
	{
	  if(t>=0)
	    {
	      gex4[i][z][2][t] = 0;
	    }
	  
	  gex3[i][z][2] = 0;
	}
      
      // update continuation values and check convergence ---------------
      double tmp0 = fmax(EV3[i][z][0],
			 pi*pow(xi[i],1.0-theta) - kappa0[i] +
			 EV3[i][z][1]);
	  
      double tmp1 = fmax(EV3[i][z][0],
			 pi*pow(xi[i],1.0-theta) - kappa1[i] +
			 rho0[i]*EV3[i][z][1] +
			 (1.0-rho0[i])*EV3[i][z][2]);

      double tmp2 = fmax(EV3[i][z][0],
			 pi - kappa1[i] +
			 (1.0-rho1[i])*EV3[i][z][1] +
			 rho1[i]*EV3[i][z][2]);

      double diff0 = fabs(tmp0-V3[i][z][0]);
      double diff1 = fabs(tmp1-V3[i][z][1]);
      double diff2 = fabs(tmp2-V3[i][z][2]);

      if(diff0>*maxdiff)
	{
	  *maxdiff=diff0;
	  imaxdiff[0]=z;
	  imaxdiff[1]=0;
	  imaxdiff[2]=0;
	}

      if(diff1>*maxdiff)
	{
	  *maxdiff=diff1;
	  imaxdiff[0]=z;
	  imaxdiff[1]=1;
	  imaxdiff[2]=0;
	}
	  
      if(diff2>*maxdiff)
	{
	  *maxdiff=diff2;
	  imaxdiff[0]=z;
	  imaxdiff[1]=2;
	  imaxdiff[2]=0;
	}

      V3[i][z][0] = tmp0;
      V3[i][z][1] = tmp1;
      V3[i][z][2] = tmp2;
    }
}

// solve policy function for industry i in temporary TPU transition period
int solve_policies_tpu(int i)
{

  for(int z=0; z<NZ; z++)
    {
      V3[i][z][0] = V2[i][z][0];
      V3[i][z][1] = V2[i][z][1];
      V3[i][z][2] = V2[i][z][2];
    }
  
  time_t start, stop;
  time(&start);

  int status = 0;
  double maxdiff = 999;
  int imaxdiff[3];
  
  int iter=0;
  int t=NU-1;
  do
    {
      iter++;
      iterate_policies_tpu(i,t,&maxdiff,imaxdiff);
      t--;

      if(verbose==3)
	printf("\t\tIter %d, diff = %0.2g\n",iter,maxdiff);
    }
  while(maxdiff>policy_tol_abs && iter < policy_max_iter);

  time(&stop);

  if(iter==policy_max_iter)
    {
      status=1;
      if(verbose==2)
	{
	  printf("\tTPU value function iteration failed for industry %d! Diff = %0.4g\n",
		 i,maxdiff);
	}
    }
  else
    {
      if(verbose==2)
	{
	  printf("\tTPU value function for industry %d converged to tolerance %0.4g in %d iterations/%0.0f seconds!\n",
		 i,maxdiff,iter,difftime(stop,start));
	}
    }

  return status;

}

// solve tpu policies for all industries in parallel
int solve_policies2_tpu()
{
  if(verbose)
    printf("\nSolving TPU policy functions\n");

  int cnt=0;
  
  time_t start, stop;
  time(&start);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<NI; i++)
    {
      if(!policy_solved_flag[i])
	{
	  policy_solved_flag[i] = solve_policies_tpu(i);
	  cnt += policy_solved_flag[i];
	}
    }

  time(&stop);
  
  if(verbose)
    {
      printf("Finished TPU policies in %0.0f seconds. %d failed to converge.\n",difftime(stop,start),cnt);
    }
  
  return 0;  
}



///////////////////////////////////////////////////////////////////////////////
// 4. Simulation
///////////////////////////////////////////////////////////////////////////////

// storage for simulated data
// we use first NT periods to simulate convergence to pre-reform steady state and NT periods for post-reform dynamics
unsigned long int seed = 0;
double z_rand[NS][NI][NF][NT*2];
double switch_rand[NS][NI][NF][NT*2];
double surv_rand[NS][NI][NF][NT*2];
double v_sim[NS][NI][NF][NT*2];

// draw random variables
void random_draws()
{
  printf("\nDrawing random numbers for simulation...\n");
  
  time_t start, stop;
  time(&start);
  
  gsl_rng_env_setup();
  gsl_rng * r = gsl_rng_alloc(gsl_rng_default);

  for(int s=0; s<NS; s++)
    {
      for(int i=0; i<NI; i++)
	{
	  for(int f=0; f<NF; f++)
	    {
	      for(int t=0; t<NT*2; t++)
		{
		  z_rand[s][i][f][t] = gsl_rng_uniform(r);
		  switch_rand[s][i][f][t] = gsl_rng_uniform(r);
		  surv_rand[s][i][f][t] = gsl_rng_uniform(r);
		}
	    }
	}
    }

  gsl_rng_free(r);

  time(&stop);
  printf("Random draws finished! Time = %0.0f\n",difftime(stop,start));
}

// main simulation function
void simul(int s, int i, int reform_flag)
{
  time_t start, stop;
  time(&start);

  gsl_interp_accel * acc1 = gsl_interp_accel_alloc();

  // for each firm in the sample...
  for(int f=0; f<NF; f++)
    {      
      // find initial value of shock based on random draw and ergodic distribution
      gsl_interp_accel_reset(acc1);
      int z = gsl_interp_accel_find(acc1, z_ucond_cumprobs[i], NZ, z_rand[s][i][f][0]);
      
      // start off as a non-exporter
      int e=0;

      // loop over the time periods in the simulation
      for(int t=0; t<NT*2; t++)
	{
	  // if the firm dies, set its exports to zero and draw a new productivity
	  if(surv_rand[s][i][f][t]>delta[i][z])
	    {
	      v_sim[s][i][f][t] = -99.9;
	      e=0;
	      
	      if(t<NT*2-1)
		z = gsl_interp_accel_find(acc1, z_ucond_cumprobs[i], NZ, z_rand[s][i][f][t+1]);
	    }
	  // if the firm survives...
	  else
	    {
	      // determine which policy function to use depending on the time period and reform flag
	      int gex_=0;
	      if(t<=NT) // pre-reform policy function used in first NT periods, plus first period after reform
		{
		  gex_ = gex[i][z][e];
		}
	      else if(t>NT) // post-reform policy function
		{
		  if(reform_flag==0) // no TPU
		    {
		      gex_ = gex2[i][z][e];
		    }
		  else if(reform_flag==1) // firms believe TPU is permanent
		    {
		      if(t-NT>NU) // once NU periods have passed, there is no TPU
			{
			  gex_ = gex2[i][z][e];
			}
		      else // before then, firms assume TPU will last forever
			{
			  gex_ = gex3[i][z][e];
			}
		    }
		  else if(reform_flag==2)
		    {
		      if(t-NT>NU)  // once NU periods have passed, there is no TPU
			{
			  gex_ = gex2[i][z][e];
			}
		      else // before then, firms know TPU will end after NU periods
			{
			  gex_ = gex4[i][z][e][t-NT-1];
			}
		    }
		}
	      

	      // determine which profit multiplier to use depending on the time period
	      double tau_hat_;
	      if(t<NT) // pre-reform multiplier used in first NT periods
		{
		  tau_hat_ = tau_hat[i];
		}
	      else if(t>=NT) // post-reform policy function used starting in first period after reform
		{
		  tau_hat_ = tau_hat2[i];
		}

	      if(gex_) // if the firm chooses to be an exporter...
		{
		  if(e==0) // if it is currently a non-exporter
		    {
		      e=1; // switch to a bad exporter
		      v_sim[s][i][f][t] = theta*theta_hat*tau_hat_*z_hat[i][z]*pow(xi[i],1.0-theta); // compute exports
		    }
		  else if (e==1) // if it is currently a bad exporter...
		    {
		      v_sim[s][i][f][t] = theta*theta_hat*tau_hat_*z_hat[i][z]*pow(xi[i],1.0-theta); // compute exports

		      // determine whether it gets to become a good exporter or stay a bad exporter
		      if(switch_rand[s][i][f][t]<rho0[i])
			{
			  e=1;
			}
		      else
			{
			  e=2;
			}
		    }
		  else if(e==2) // if it is current a good exporter...
		    {
		      v_sim[s][i][f][t] = theta*theta_hat*tau_hat_*z_hat[i][z]; // compute exports

		      // determine whether it gets to stay a good exporter or switched back to a bad one
		      if(switch_rand[s][i][f][t]<rho1[i])
			{
			  e=2;
			}
		      else
			{
			  e=1;
			}
		    }
		}
	      else // if the firm chooses not to be an exporter, set its exports to zero
		{
		  e=0;
		  v_sim[s][i][f][t] = -99.9;
		}

	      // draw a productivity shock
	      if(t<NT*2-1)
		z = gsl_interp_accel_find(acc1, z_trans_cumprobs[i][z], NZ, z_rand[s][i][f][t+1]);

	    }
	}      
    }

  gsl_interp_accel_free(acc1);

  time(&stop);

  if(verbose==2)
    printf("\tSimulation %d completed for industry %s in %0.0f seconds.\n",s,industry[i],difftime(stop,start));

  return;
}

// do all simulations in parallel
void simul2(int reform_flag,
	    double *avg_expart_rate,
	    double *avg_exit_rate,
	    double *avg_new_size,
	    int calc_moments)
{
  if(verbose && reform_flag==0)
    printf("\nDoing simulations without TPU...\n");
  else if(verbose && reform_flag==1)
    printf("\nDoing simulations with permanent TPU...\n");
  else if(verbose && reform_flag==2)
    printf("\nDoing simulations with %d-period transition with TPU...\n",NU);

  time_t start, stop;
  time(&start);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int s=0; s<NS; s++)
    {
      for(int i = 0; i<NI; i++)
	{
	  if(policy_solved_flag[i]==0)
	    {
	      simul(s,i,reform_flag);
	    }
	}
    }

  if(calc_moments)
    {
      // compute average export participation rate, exit rate, and new exporter size across all simulations in steady state
      int t=NT-1;
      *avg_expart_rate=0.0;
      *avg_exit_rate=0.0;
      *avg_new_size=0.0;
      int num_sims_exit=0;
      int num_sims_newsize=0;
  
      for(int s=0; s<NS; s++)
	{
	  int num_exporters=0;
	  int num_firms=0;
	  int num_exits=0;
	  int num_new_exporters=0;
	  int num_incumbents=0;

	  double newsize=0.0;
	  double incsize=0.0;
      
	  for(int i=0; i<NI; i++)
	    {
	      if(policy_solved_flag[i]==0)
		{
		  num_firms += NF;
		  for(int f=0; f<NF; f++)
		    {
		      if(v_sim[s][i][f][t]>1.0e-10)
			{
			  num_exporters += 1;
		      
			  if(v_sim[s][i][f][t+1]<0.0)
			    {
			      num_exits += 1;
			    }

			  if(v_sim[s][i][f][t-1]<0.0)
			    {
			      num_new_exporters += 1;
			      newsize += v_sim[s][i][f][t];
			    }
			  else
			    {
			      num_incumbents += 1;
			      incsize += v_sim[s][i][f][t];
			    }
			}
		    }
		}
	    }
	  *avg_expart_rate += ((double)(num_exporters))/((double)(num_firms));
      
	  if(num_exporters>0)
	    {
	      *avg_exit_rate += ((double)(num_exits))/((double)(num_exporters));
	      num_sims_exit += 1;
	    }

	  if(num_incumbents>0 && num_new_exporters>0)
	    {
	      *avg_new_size += (newsize/((double)(num_new_exporters))) / (incsize/((double)(num_incumbents)));
	      num_sims_newsize += 1;
	    }

	}
  
      *avg_expart_rate = *avg_expart_rate/((double)(NS));
      *avg_exit_rate = *avg_exit_rate/((double)(num_sims_exit));
      *avg_new_size  = *avg_new_size/((double)(num_sims_newsize));
    }

	
  time(&stop);

  if(verbose)
    {
      printf("Finished simulations in %0.0f seconds.\n",difftime(stop,start));
      if(calc_moments)
	{
	  printf("\tExport part. rate = %0.8f\n\tExit rate = %0.8f\n\tNew exporter relative size = %0.8f\n",
		 100*(*avg_expart_rate),100*(*avg_exit_rate),*avg_new_size);
	}
    }
    
  return;
}

// create panel dataset from simulation
void create_panel_dataset(int reform_flag)
{
  if(verbose && reform_flag==0)
    printf("Creating panel dataset from simulation without TPU...\n");
  else if(verbose && reform_flag==1)
    printf("Creating panel dataset from simulation with permanent TPU...\n");
  else if(verbose && reform_flag==2)
    printf("Creating panel dataset from simulation %d-period transition with TPU ...\n",NU);

  time_t start, stop;
  time(&start);

  FILE * file = 0x0;
  if(reform_flag==0)
    file = fopen("output/simul_no_tpu.csv","w");
  else if(reform_flag==1)
    file = fopen("output/simul_tpu.csv","w");
  else if(reform_flag==2)
    file = fopen("output/simul_tpu2.csv","w");

  fprintf(file,"s,i,y,f,tau,tpu_exposure,v,entrant,incumbent,exit\n");
  for(int s=0; s<NS; s++)
    {
      for(int i=0; i<NI; i++)
	{
	  for(int f=0; f<NF; f++)
	    {
	      for(int t=NT-1; t<NT*2; t++)
		{
		  if(policy_solved_flag[i]==0 && v_sim[s][i][f][t]>1.0e-10)
		    {
		      int exit = v_sim[s][i][f][t+1]>1.0e-10;
		      int entrant = v_sim[s][i][f][t-1]<0.0;
		      int incumbent = 1-entrant;
		      double tau_ = t>=NT ? tau2[i] : tau[i];
		      fprintf(file,"%d,%s,%d,FIRM%d,%0.16f,%0.16f,%0.16f,%d,%d,%d\n",
			      s,industry[i],t-(NT-1),f,tau_,tau[i]-tau2[i],v_sim[s][i][f][t],exit,entrant,incumbent);
		    }
		}
	    }
	}
    }

  fclose(file);

  time(&stop);

  if(verbose)
    printf("Panel data construction complete in %0.0f seconds.\n",difftime(stop,start));
}

///////////////////////////////////////////////////////////////////////////////
// 5. Main function and setup/cleanup
///////////////////////////////////////////////////////////////////////////////

int setup()
{
  printf("Setting up model environment...\n");
    
  time_t start, stop;
  time(&start);

#ifdef _OPENMP
  printf("Parallel processing with %d threads\n",omp_get_max_threads());
#else
  printf("No parallelization\n");
#endif

  if(init_params())
    {
      printf("Failed to initialize parameters!\n");
      return 1;
    }

  for(int i=0; i<NI; i++)
    {
      discretize_z(i);
      calc_death_probs(i);
    }

  random_draws();

  time(&stop);
  printf("\nSetup complete! Runtime = %0.0f seconds.\n",
	 difftime(stop,start));
	  
  return 0;
}

int analysis()
{
  printf("Solving and simulating model...\n");

  time_t start, stop;
  time(&start);

  if(solve_policies2())
    return 1;

  if(solve_policies2_tpu())
    return 1; 

  double expart_rate=0.0;
  double exit_rate=0.0;
  double new_size;
  simul2(0,&expart_rate,&exit_rate,&new_size,1);
  create_panel_dataset(0);

  simul2(1,&expart_rate,&exit_rate,&new_size,0);
  create_panel_dataset(1);

  simul2(2,&expart_rate,&exit_rate,&new_size,0);
  create_panel_dataset(2);

  time(&stop);
  printf("\nAnalysis complete! Runtime = %0.0f seconds.\n",
	 difftime(stop,start));
	  
  return 0;
}

int main()
{
  time_t start, stop;
  time(&start);

  // setup environment
  linebreak();    
  if(setup())
      return 1;

  // solve and simulate model
  linebreak();	  
  if(analysis())
      return 1;

  // finish program
  linebreak();  
  time(&stop);
  printf("\nProgram complete! Total runtime: %0.16f seconds.\n",difftime(stop,start));

  return 0;
}


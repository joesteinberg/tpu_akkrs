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
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_types.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_interp.h>
#include <omp.h>

// macros: discretization
#define NI 34 // number of industries
#define NZ 201 // productivity shock grid size
#define NT 100 // simulation length
#define NS 100 // number of simulations
#define NF 2000 // simulation population size

// macros: paralellization
#ifdef _OPENMP
#define PAR 1
#else
#define PAR 0
#endif


// hardcoded params
// time starts in 1971
const int t_reform = 9; // = 1980 - 1971
const int t_wto = 30; // = 2001 - 1971
const int t_data_max = 38; // = 2009 - 1971
const int root_max_iter = 100;
const double root_tol_rel = 1.0e-6;
const double root_tol_abs = 1.0e-6;
const double delta_deriv = 1.0e-9;
const double delta_root = 1.0e-9;
const double policy_tol_abs = 1.0e-10;
const int policy_max_iter = 2000;
const double x_grid_ub_mult = 10.0;
const double x_grid_exp = 1.0;
const double tpu_prob_update_speed = 0.01;
const double coeff_err_tol = 1.0e-3;
const int max_cal_iter = 100;

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

void expspace(double lo, double hi, int n, double ex, double * v)
{
  linspace(0.0,pow(hi-lo,1.0/ex),n,v);
  int i;
  for(i=0;i<n;i++)
    {
      v[i] = pow(v[i],ex)+lo;
    }
  return;
}

void reverse(double * v, int n)
{
  double * tmp = (double *)malloc(n*sizeof(double));
  memcpy(tmp,v,n*sizeof(double));
  int i;
  for(i=0; i<n; i++)
    {
      v[i]=tmp[n-1-i];
    }
  free(tmp);
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
double tau_applied[NI][NT] = {{0.0}}; // trade cost before liberaliztion
double tau_nntr[NI][NT] = {{0.0}}; // trade cost before liberaliztion
//double pi_hat[NI] = {0.0}; // theta_hat*tau_hat
//double tau2[NI] = {0.0}; // trade cost after liberalization
//double tau_hat2[NI] = {0.0}; // = tau2^(1-theta)
//double pi_hat2[NI] = {0.0}; // theta_hat*tau_hat2
double tpu_prob_temp[NT] = {0.0}; // probability of reform reverting
double tpu_prob_perm[NT] = {0.0}; // probability of reform reverting
  
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
  Q = 0.96;
  theta = 4;
  theta_hat = (1.0/theta) * pow(theta/(theta-1.0),1.0-theta);

  for(int i=0; i<NI; i++)
    {
      delta0[i] = 21.04284098;
      delta1[i] = 0.02258301;
      sig_z[i] = 1.32;
      rho_z[i] =  0.65;

      // theta=4
      kappa0[i] = 0.24;
      kappa1[i] = 0.18;
      xi[i]=2.00541880064;
      
      // theta=4.5
      //kappa0[i] = 0.2;
      //kappa1[i] = 0.155;

      // theta=5
      //kappa0[i] = 0.175;
      //kappa1[i] = 0.135;
      //xi[i]=1.60541880064;
      mu_e[i] = 1.34;
      //xi[i] = 1.5;
      rho0[i]=0.91571120;
      rho1[i]=0.91571120;
    }


  FILE * file = fopen("output/tpuprobs.txt","r");
  if(!file)
    {
      printf("Failed to open file with TPU prob guesses!\n");
      return 1;
    }
  else
    {
      int got = 0;
      for(int t=0; t<21; t++)
	{
	  got += fscanf(file,"%lf",&(tpu_prob_temp[t+t_reform]));
	}
      for(int t=0; t<21; t++)
	{
	  got += fscanf(file,"%lf",&(tpu_prob_perm[t+t_reform]));
	}
      fclose(file);
      if(got != 42)
	{
	  printf("Failed to load TPU prob guesses!\n");
	  return 1;
	}
    }
  //expspace(0.04, 0.19, 21, 0.4, &(tpu_prob_temp[t_reform]));
  //expspace(0.04, 0.19, 21, 0.4, &(tpu_prob_perm[t_reform]));
  //reverse(&(tpu_prob_temp[t_reform]),21);
  //reverse(&(tpu_prob_perm[t_reform]),21);
  
  // set all destination-specific variables to mean values... we will use the
  // array of destinations in parallelizing the calibration
  file = fopen("../scripts/tariff_data2.csv","r");
  if(!file)
    {
      printf("Failed to open file with tariff data!\n");
      return 1;
    }
  else
    {
      char buffer[128];
      int t;
      double tau_, nntr_;
      int i;

      while(fscanf(file,"%d %s %d %lf %lf",&i,buffer,&t,&tau_,&nntr_) == 5)
      {
	tau_applied[i][t] = 1.0 + tau_;
	tau_nntr[i][t] = 1.0 + nntr_;
	strncpy(industry[i],buffer,128);
      }
      if(!feof(file))
	{
	  printf("Failed to load data!\n");
	  fclose(file);
	  return 1;
	}
      fclose(file);

      for(int i=0; i<NI; i++)
	{
	  for(int t=0; t<3; t++)
	    {
	      tau_applied[i][t] = tau_applied[i][3];
	      tau_nntr[i][t] = tau_nntr[i][3];
	    }
	}
      
      for(int t=t_data_max+1; t<NT; t++)
	{
	  for(int i=0; i<NI; i++)
	    {
	      tau_applied[i][t] = tau_applied[i][t-1];
	      tau_nntr[i][t] = tau_nntr[i][t-1];
	    }
	}

      /*
      for(int i=0; i<NI; i++)
	{
	  for(int t=0; t<NT; t++)
	    {
	      printf("%.3s %d %f %f\n",industry[i],t,tau_applied[i][t],tau_nntr[i][t]);
	    }
	    }*/
      
      return 0;
    }
}

///////////////////////////////////////////////////////////////////////////////
// 3. Dynamic program
///////////////////////////////////////////////////////////////////////////////

double V_pre80[NI][NZ][3] = {{{0.0}}}; // pre-1980 value function
double V_nntr[NI][NZ][3] = {{{0.0}}}; // NNTR value function
double V_ref_det[NI][NZ][3][NT] = {{{{0.0}}}}; // post-1980 value function without TPU
double V_ref_tpu_perm[NI][NZ][3][NT] = {{{{0.0}}}}; // post-1980 value function with permanent TPU
double V_ref_tpu_temp[NI][NZ][3][NT] = {{{{0.0}}}}; // post-1980 value function with temporary TPU

double EV_pre80[NI][NZ][3] = {{{0.0}}}; // pre-1980 continuation value
double EV_nntr[NI][NZ][3] = {{{0.0}}}; // NNTR continuation value
double EV_ref_det[NI][NZ][3] = {{{0.0}}}; // post-1980 continuation value without TPU
double EV_ref_tpu_perm[NI][NZ][3] = {{{0.0}}}; // post-1980 continuation value with perm. TPU
double EV_ref_tpu_temp[NI][NZ][3] = {{{0.0}}}; // post-1980 continuation value with temp. TPU

int gex_pre80[NI][NZ][3] = {{{0}}}; // pre-1980 policy function
int gex_nntr[NI][NZ][3] = {{{0}}}; // NNTR policy function
int gex_ref_det[NI][NZ][3][NT] = {{{{0}}}}; // post-1980 policy function without TPU
int gex_ref_tpu_perm[NI][NZ][3][NT] = {{{{0}}}}; // post-1980 policy function with perm. TPU
int gex_ref_tpu_temp[NI][NZ][3][NT] = {{{{0}}}}; // post-1980 policy function with temp. TPU

int policy_solved_flag[NI] = {0};

// initial guess for value functions
void init_dp_objs()
{
  for(int i=0; i<NI; i++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  double pi_hat = theta_hat * pow(tau_applied[i][0],1.0-theta);
	  V_pre80[i][iz][0] = 0.0;
	  V_pre80[i][iz][1] = pi_hat*pow(xi[i],1.0-theta)*z_hat[i][iz]/Q;
	  V_pre80[i][iz][2] = pi_hat*z_hat[i][iz]/Q;
	  
	  pi_hat = theta_hat * pow(tau_nntr[i][0],1.0-theta);
	  V_nntr[i][iz][0] = 0.0;
	  V_nntr[i][iz][1] = pi_hat*pow(xi[i],1.0-theta)*z_hat[i][iz]/Q;
	  V_nntr[i][iz][2] = pi_hat*z_hat[i][iz]/Q;
	  
	  for(int t=0; t<NT; t++)
	    {
	      pi_hat = theta_hat * pow(tau_applied[i][t],1.0-theta);
	      V_ref_det[i][iz][0][t] = 0.0;
	      V_ref_det[i][iz][1][t] = pi_hat*pow(xi[i],1.0-theta)*z_hat[i][iz]/Q;
	      V_ref_det[i][iz][2][t] = pi_hat*z_hat[i][iz]/Q;
	    }
	}
    }
}

// steady state Bellman
void calc_EV_ss(int i)
{
    for(int z=0; z<NZ; z++)
    {
      for(int e=0; e<3; e++)
	{
	  EV_pre80[i][z][e]=0.0;
	  EV_nntr[i][z][e]=0.0;
	  for(int zp=0; zp<NZ; zp++)
	    {
	      if(z_trans_probs[i][z][zp]>1.0e-11)
		{
		  EV_pre80[i][z][e] += Q*delta[i][z]*V_pre80[i][zp][e]*z_trans_probs[i][z][zp];
		  EV_nntr[i][z][e] += Q*delta[i][z]*V_nntr[i][zp][e]*z_trans_probs[i][z][zp];
		}
	    }
	} 
    }

}

void iterate_policies_ss(int i, double * maxdiff, int imaxdiff[3])
{
  *maxdiff=-HUGE_VAL;

  calc_EV_ss(i);
  
  for(int z=0; z<NZ; z++)
    {
      // first compute policy functions -------------------
      
      // nntr
      if(EV_nntr[i][z][0] < EV_nntr[i][z][1] - kappa0[i])
	{
	  gex_nntr[i][z][0] = 1;
	}
      else
	{
	  gex_nntr[i][z][0] = 0;
	}

      if(EV_nntr[i][z][0] <
	 rho0[i]*EV_nntr[i][z][1] + (1.0-rho0[i])*EV_nntr[i][z][2] - kappa1[i])
	{
	  gex_nntr[i][z][1] = 1;
	}
      else
	{
	  gex_nntr[i][z][1] = 0;
	}
	  
      if(EV_nntr[i][z][0] <
	 (1.0-rho1[i])*EV_nntr[i][z][1] + rho1[i]*EV_nntr[i][z][2] - kappa1[i])
	{
	  gex_nntr[i][z][2] = 1;
	}
      else
	{
	  gex_nntr[i][z][2] = 0;
	}

      // applied
      if(EV_pre80[i][z][0] < EV_pre80[i][z][1] - kappa0[i])
	{
	  gex_pre80[i][z][0] = 1;
	}
      else
	{
	  gex_pre80[i][z][0] = 0;
	}

      if(EV_pre80[i][z][0] <
	 rho0[i]*EV_pre80[i][z][1] + (1.0-rho0[i])*EV_pre80[i][z][2] - kappa1[i])
	{
	  gex_pre80[i][z][1] = 1;
	}
      else
	{
	  gex_pre80[i][z][1] = 0;
	}
	  
      if(EV_pre80[i][z][0] <
	 (1.0-rho1[i])*EV_pre80[i][z][1] + rho1[i]*EV_pre80[i][z][2] - kappa1[i])
	{
	  gex_pre80[i][z][2] = 1;
	}
      else
	{
	  gex_pre80[i][z][2] = 0;
	}
      
      // update continuation values and check convergence ---------------
      // pre-reform
      double pi = theta_hat * pow(tau_nntr[i][0],1.0-theta) *z_hat[i][z];     
      double tmp0 = fmax(EV_nntr[i][z][0], EV_nntr[i][z][1] - kappa0[i]);
      
      double tmp1 = pi*pow(xi[i],1.0-theta) +
	fmax(EV_nntr[i][z][0],
	     rho0[i]*EV_nntr[i][z][1] + (1.0-rho0[i])*EV_nntr[i][z][2] - kappa1[i]);

      double tmp2 = pi +
	fmax(EV_nntr[i][z][0],
	     (1.0-rho1[i])*EV_nntr[i][z][1] + rho1[i]*EV_nntr[i][z][2] - kappa1[i]);
	  
      double diff0 = fabs(tmp0-V_nntr[i][z][0]);
      double diff1 = fabs(tmp1-V_nntr[i][z][1]);
      double diff2 = fabs(tmp2-V_nntr[i][z][2]);

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

      V_nntr[i][z][0] = tmp0;
      V_nntr[i][z][1] = tmp1;
      V_nntr[i][z][2] = tmp2;

      // post-reform
      pi = theta_hat * pow(tau_applied[i][0],1.0-theta) *z_hat[i][z];
      tmp0 = fmax(EV_pre80[i][z][0], EV_pre80[i][z][1] - kappa0[i]);
      
      tmp1 = pi*pow(xi[i],1.0-theta) +
	fmax(EV_pre80[i][z][0],
	     rho0[i]*EV_pre80[i][z][1] + (1.0-rho0[i])*EV_pre80[i][z][2] - kappa1[i]);

      tmp2 = pi +
	fmax(EV_pre80[i][z][0],
	     (1.0-rho1[i])*EV_pre80[i][z][1] + rho1[i]*EV_pre80[i][z][2] - kappa1[i]);
	  
      diff0 = fabs(tmp0-V_pre80[i][z][0]);
      diff1 = fabs(tmp1-V_pre80[i][z][1]);
      diff2 = fabs(tmp2-V_pre80[i][z][2]);

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

      V_pre80[i][z][0] = tmp0;
      V_pre80[i][z][1] = tmp1;
      V_pre80[i][z][2] = tmp2;
    }
}

// deterministic Bellman
void calc_EV_det(int i, int t)
{
  for(int z=0; z<NZ; z++)
    {
      for(int e=0; e<3; e++)
	{
	  EV_ref_det[i][z][e]=0.0;
	  for(int zp=0; zp<NZ; zp++)
	    {
	      if(z_trans_probs[i][z][zp]>1.0e-11)
		{
		  EV_ref_det[i][z][e] += Q*delta[i][z]*V_ref_det[i][zp][e][t]*z_trans_probs[i][z][zp];
		}
	    }
	} 
    }
  
}

void iterate_policies_det(int i, int t, double * maxdiff, int imaxdiff[3])
{
  *maxdiff=-HUGE_VAL;

  if(t<NT-1)
    calc_EV_det(i,t+1);
  else
    calc_EV_det(i,t);
  
  for(int z=0; z<NZ; z++)
    {
      // applied
      if(EV_ref_det[i][z][0] < EV_ref_det[i][z][1] - kappa0[i])
	{
	  gex_ref_det[i][z][0][t] = 1;
	}
      else
	{
	  gex_ref_det[i][z][0][t] = 0;
	}

      if(EV_ref_det[i][z][0] <
	 rho0[i]*EV_ref_det[i][z][1] + (1.0-rho0[i])*EV_ref_det[i][z][2] - kappa1[i])
	{
	  gex_ref_det[i][z][1][t] = 1;
	}
      else
	{
	  gex_ref_det[i][z][1][t] = 0;
	}
	  
      if(EV_ref_det[i][z][0] <
	 (1.0-rho1[i])*EV_ref_det[i][z][1] + rho1[i]*EV_ref_det[i][z][2] - kappa1[i])
	{
	  gex_ref_det[i][z][2][t] = 1;
	}
      else
	{
	  gex_ref_det[i][z][2][t] = 0;
	}
      
      // update continuation values and check convergence ---------------
      double pi = theta_hat * pow(tau_applied[i][t],1.0-theta) *z_hat[i][z];
      double tmp0 = fmax(EV_ref_det[i][z][0], EV_ref_det[i][z][1] - kappa0[i]);
      
      double tmp1 = pi*pow(xi[i],1.0-theta) +
	fmax(EV_ref_det[i][z][0],
	     rho0[i]*EV_ref_det[i][z][1] + (1.0-rho0[i])*EV_ref_det[i][z][2] - kappa1[i]);

      double tmp2 = pi +
	fmax(EV_ref_det[i][z][0],
	     (1.0-rho1[i])*EV_ref_det[i][z][1] + rho1[i]*EV_ref_det[i][z][2] - kappa1[i]);
	  
      double diff0 = fabs(tmp0-V_ref_det[i][z][0][t]);
      double diff1 = fabs(tmp1-V_ref_det[i][z][1][t]);
      double diff2 = fabs(tmp2-V_ref_det[i][z][2][t]);

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

      V_ref_det[i][z][0][t] = tmp0;
      V_ref_det[i][z][1][t] = tmp1;
      V_ref_det[i][z][2][t] = tmp2;
    }
}

// solve policy function for industry i
int solve_policies(int i)
{
  init_dp_objs(i);

  double maxdiff = 999;
  int imaxdiff[3];

  // first do SS policies for pre-1980 and NNTR rates
  int iter=0;
  do
    {
      iter++;
      iterate_policies_ss(i,&maxdiff,imaxdiff);
    }
  while(maxdiff>policy_tol_abs && iter < policy_max_iter);

  if(iter==policy_max_iter)
    {
      printf("\tValue function iteration failed for industry %d! Diff = %0.4g\n",
	     i,maxdiff);
      return 1;
    }

  // now do SS policy for very last period of post-1980 period
  do
    {
      iter++;
      iterate_policies_det(i,NT-1,&maxdiff,imaxdiff);
    }
  while(maxdiff>policy_tol_abs && iter < policy_max_iter);

  if(iter==policy_max_iter)
    {
      printf("\tValue function iteration failed for industry %d! Diff = %0.4g\n",
	     i,maxdiff);
      return 1;
    }

  // now iterate backwards
  for(int t=NT-2; t>=t_reform; t--)
    {
      iterate_policies_det(i,t,&maxdiff,imaxdiff);
    }

  return 0;

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
void calc_EV_tpu_temp(int i, int t)
{
  for(int z=0; z<NZ; z++)
    {
      for(int e=0; e<3; e++)
	{
	  EV_ref_tpu_temp[i][z][e]=0.0;
	  for(int zp=0; zp<NZ; zp++)
	    {
	      if(z_trans_probs[i][z][zp]>1.0e-11)
		{
		  EV_ref_tpu_temp[i][z][e] += Q*delta[i][z]*z_trans_probs[i][z][zp] *
		    (tpu_prob_temp[t]*V_nntr[i][zp][e] + (1.0-tpu_prob_temp[t])*V_ref_tpu_temp[i][zp][e][t]);
		}
	    }
	} 
    }
  
}

void iterate_policies_tpu_temp(int i, int t)
{
  if(t<NT-1)
    calc_EV_tpu_temp(i,t+1);
  else
    calc_EV_tpu_temp(i,t);
  
  for(int z=0; z<NZ; z++)
    {
      // first compute policy functions -------------------
      if(EV_ref_tpu_temp[i][z][0]< EV_ref_tpu_temp[i][z][1] - kappa0[i])
	{
	  gex_ref_tpu_temp[i][z][0][t] = 1;
	}
      else
	{  
	  gex_ref_tpu_temp[i][z][0][t] = 0;
	}

      if(EV_ref_tpu_temp[i][z][0] <
	 rho0[i]*EV_ref_tpu_temp[i][z][1] + (1.0-rho0[i])*EV_ref_tpu_temp[i][z][2] - kappa1[i])
	{ 
	  gex_ref_tpu_temp[i][z][1][t] = 1;
	}
      else
	{
	  gex_ref_tpu_temp[i][z][1][t] = 0;
	}
	  
      if(EV_ref_tpu_temp[i][z][0]<
	 (1.0-rho1[i])*EV_ref_tpu_temp[i][z][1] + rho1[i]*EV_ref_tpu_temp[i][z][2] - kappa1[i])
	{
	  gex_ref_tpu_temp[i][z][2][t] = 1;
	}
      else
	{  
	  gex_ref_tpu_temp[i][z][2][t] = 0;
	}
      
      // update continuation values and check convergence ---------------
      double pi = theta_hat * pow(tau_applied[i][t],1.0-theta) *z_hat[i][z];
      
      double tmp0 = fmax(EV_ref_tpu_temp[i][z][0], EV_ref_tpu_temp[i][z][1] - kappa0[i]);
	  
      double tmp1 = pi*pow(xi[i],1.0-theta) +
	fmax(EV_ref_tpu_temp[i][z][0],
	     rho0[i]*EV_ref_tpu_temp[i][z][1] + (1.0-rho0[i])*EV_ref_tpu_temp[i][z][2] - kappa0[i]);

      double tmp2 = pi +
	fmax(EV_ref_tpu_temp[i][z][0],		       
	     (1.0-rho1[i])*EV_ref_tpu_temp[i][z][1] + rho1[i]*EV_ref_tpu_temp[i][z][2] - kappa0[i]);

      V_ref_tpu_temp[i][z][0][t] = tmp0;
      V_ref_tpu_temp[i][z][1][t] = tmp1;
      V_ref_tpu_temp[i][z][2][t] = tmp2;
    }
}

// Bellman equation iteration for permanent TPU transition
void calc_EV_tpu_perm(int i, int t, double prob)
{
  for(int z=0; z<NZ; z++)
    {
      for(int e=0; e<3; e++)
	{
	  EV_ref_tpu_perm[i][z][e]=0.0;
	  for(int zp=0; zp<NZ; zp++)
	    {
	      if(z_trans_probs[i][z][zp]>1.0e-11)
		{
		  EV_ref_tpu_perm[i][z][e] += Q*delta[i][z]*z_trans_probs[i][z][zp] *
		  (prob*V_nntr[i][zp][e] + (1.0-prob)*V_ref_tpu_perm[i][zp][e][t]);
		}
	    }
	} 
    }
  
}

void iterate_policies_tpu_perm(int i, int t, double prob, double * maxdiff, int imaxdiff[3])
{
  *maxdiff=-HUGE_VAL;

  if(t<NT-1)
    calc_EV_tpu_perm(i,t+1,prob);
  else
    calc_EV_tpu_perm(i,t,prob);

  for(int z=0; z<NZ; z++)
    {
      // first compute policy functions -------------------
      if(EV_ref_tpu_perm[i][z][0]< EV_ref_tpu_perm[i][z][1] - kappa0[i])
	{
	  gex_ref_tpu_perm[i][z][0][t] = 1;
	}
      else
	{  
	  gex_ref_tpu_perm[i][z][0][t] = 0;
	}

      if(EV_ref_tpu_perm[i][z][0] <
	 rho0[i]*EV_ref_tpu_perm[i][z][1] + (1.0-rho0[i])*EV_ref_tpu_perm[i][z][2] - kappa1[i])
	{ 
	  gex_ref_tpu_perm[i][z][1][t] = 1;
	}
      else
	{
	  gex_ref_tpu_perm[i][z][1][t] = 0;
	}
	  
      if(EV_ref_tpu_perm[i][z][0]<
	 (1.0-rho1[i])*EV_ref_tpu_perm[i][z][1] + rho1[i]*EV_ref_tpu_perm[i][z][2] - kappa1[i])
	{
	  gex_ref_tpu_perm[i][z][2][t] = 1;
	}
      else
	{  
	  gex_ref_tpu_perm[i][z][2][t] = 0;
	}

      
      // update continuation values and check convergence ---------------
      double pi = theta_hat * pow(tau_applied[i][t],1.0-theta) *z_hat[i][z];
     
      double tmp0 = fmax(EV_ref_tpu_perm[i][z][0], EV_ref_tpu_perm[i][z][1] - kappa0[i]);
	  
      double tmp1 = pi*pow(xi[i],1.0-theta) +
	fmax(EV_ref_tpu_perm[i][z][0],
	     rho0[i]*EV_ref_tpu_perm[i][z][1] + (1.0-rho0[i])*EV_ref_tpu_perm[i][z][2] - kappa0[i]);

      double tmp2 = pi +
	fmax(EV_ref_tpu_perm[i][z][0],		       
	     (1.0-rho1[i])*EV_ref_tpu_perm[i][z][1] + rho1[i]*EV_ref_tpu_perm[i][z][2] - kappa0[i]);
      
      double diff0 = fabs(tmp0-V_ref_tpu_perm[i][z][0][t]);
      double diff1 = fabs(tmp1-V_ref_tpu_perm[i][z][1][t]);
      double diff2 = fabs(tmp2-V_ref_tpu_perm[i][z][2][t]);

      V_ref_tpu_perm[i][z][0][t] = tmp0;
      V_ref_tpu_perm[i][z][1][t] = tmp1;
      V_ref_tpu_perm[i][z][2][t] = tmp2;

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
    }
}

// solve policy function for industry i in temporary TPU transition period
int solve_policies_tpu(int i)
{ 
  // temporary TPU value function: no need to converge anything, just iterate a few times
  for(int z=0; z<NZ; z++)
    {
      V_ref_tpu_temp[i][z][0][t_wto] = V_ref_det[i][z][0][t_wto];
      V_ref_tpu_temp[i][z][1][t_wto] = V_ref_det[i][z][1][t_wto];
      V_ref_tpu_temp[i][z][2][t_wto] = V_ref_det[i][z][2][t_wto];
    }
  
  for(int t=t_wto-1; t>=t_reform; t--)
    {
      iterate_policies_tpu_temp(i,t);
    }

  // permanent TPU is more complicated...
  /// for each period t until WTO accession...
  for(int t=t_reform; t<t_wto; t++)
    {
      // step 1: converge on the value function in the last period, assuming that reversion prob remains
      // tpu_prob[t] forever
      for(int z=0; z<NZ; z++)
	{
	  V_ref_tpu_perm[i][z][0][NT-1] = V_ref_det[i][z][0][NT-1];
	  V_ref_tpu_perm[i][z][1][NT-1] = V_ref_det[i][z][1][NT-1];
	  V_ref_tpu_perm[i][z][2][NT-1] = V_ref_det[i][z][2][NT-1];
	}
      
      double maxdiff = 999;
      int imaxdiff[3];
      
      int iter=0;
      do
	{
	  iter++;
	  iterate_policies_tpu_perm(i,NT-1,tpu_prob_perm[t],&maxdiff,imaxdiff);
	}
      while(maxdiff>policy_tol_abs && iter < policy_max_iter);

      if(iter==policy_max_iter)
	{
	  printf("\tPerm TPU value function %d iteration failed for industry %d! Diff = %0.4g\n",
		 t,i,maxdiff);
	  return 1;
	}

      // step 2: iterate backwards until we get to the current period, again applying the current period's reversion
      // probability in all future periods
      for(int s=NT-2; s>=t; s--)
	{
	  iterate_policies_tpu_perm(i,s,tpu_prob_perm[t],&maxdiff,imaxdiff);
	}
    }

  return 0;

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
double z_rand[NS][NI][NF][NT];
double switch_rand[NS][NI][NF][NT];
double surv_rand[NS][NI][NF][NT];
double v_sim[NS][NI][NF][NT];

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
	      for(int t=0; t<NT; t++)
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
      for(int t=0; t<NT; t++)
	{
	  // determine which profit multiplier to use depending on the time period
	  double tau_hat_ = pow(tau_applied[i][t],1.0-theta);
	  
	  if(e==0) // if it is currently a non-exporter
	    {
	      v_sim[s][i][f][t] = -99.9;
	    }
	  else if (e==1) // if it is currently a bad exporter...
	    {
	      v_sim[s][i][f][t] = theta*theta_hat*tau_hat_*z_hat[i][z]*pow(xi[i],1.0-theta); // compute exports
	    }
	  else if(e==2) // if it is current a good exporter...
	    {
	      v_sim[s][i][f][t] = theta*theta_hat*tau_hat_*z_hat[i][z]; // compute exports
	    }

	  if(gsl_isinf(v_sim[s][i][f][t]) || gsl_isnan(v_sim[s][i][f][t]))
	    {
	      printf("Error! Inf/Nan exports!\n");
	      printf("tau_hat = %0.6f\n",tau_hat_);
	      return;
	    }

	  // determine which policy function to use depending on the time period and reform flag
	  int gex_=0;
	  if(t>=t_wto) // if we are after wto accession, always use the deterministic policy function
	    {
	      gex_ = gex_ref_det[i][z][e][t];
	    }
	  else if(t<t_reform) // if we are before the 1980 reform, use the pre-MIT shock policy
	    {
	      gex_ = gex_pre80[i][z][e];
	    }
	  else
	    {// otherwise it depends on which scenarion we are using
	      if(reform_flag==0) // no TPU
		{
		  gex_ = gex_ref_det[i][z][e][t];
		}
	      else if(reform_flag==1) // firms believe TPU is temporary
		{
		  gex_ = gex_ref_tpu_temp[i][z][e][t];
		}
	      else if(reform_flag==2) // firms believe TPU is permanent
		{
		  gex_ = gex_ref_tpu_perm[i][z][e][t];
		}
	    }

	  // if the firm dies, exit and draw a new productivity
	  if(surv_rand[s][i][f][t]>delta[i][z])
	    {
	      e=0;
	      
	      if(t<NT-1)
		z = gsl_interp_accel_find(acc1, z_ucond_cumprobs[i],
					  NZ, z_rand[s][i][f][t+1]);
	    }
	  else
	    {
	      if(t<NT-1)
		z = gsl_interp_accel_find(acc1, z_trans_cumprobs[i][z],
					  NZ, z_rand[s][i][f][t+1]);
	      
	      // if firm decides not to export, then it exits
	      if(gex_==0)
		{
		  e=0;
		}
	      else if(gex_==1)
		{
		  if(e==0)
		    {
		      e=1;
		    }
		  else
		    {
		      if(e==1)
			{
			  if(switch_rand[s][i][f][t]<rho0[i])
			    {
			      e=1;
			    }
			  else
			    {
			      e=2;
			    }
			}
		      else if(e==2)
			{
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
		}
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
    printf("\nDoing simulations with temporary TPU...\n");
  else if(verbose && reform_flag==2)
    printf("\nDoing simulations with permanent TPU...\n");

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
      int t=NT-2;
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
	  //int num_incumbents=0;

	  double newsize=0.0;
	  double meansize=0.0;
      
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
			  meansize += v_sim[s][i][f][t];
		      
			  if(v_sim[s][i][f][t+1]<0.0)
			    {
			      num_exits += 1;
			    }

			  if(v_sim[s][i][f][t-1]<0.0)
			    {
			      num_new_exporters += 1;
			      newsize += v_sim[s][i][f][t];
			    }
			  /*else
			    {
			      num_incumbents += 1;
			      incsize += v_sim[s][i][f][t];
			      }*/
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

	  //if(num_incumbents>0 && num_new_exporters>0)
	  if(num_new_exporters>0)
	    {
	      *avg_new_size += (newsize/((double)(num_new_exporters))) / (meansize/((double)(num_exporters)));
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
	  printf("\tExport part. rate = %0.8f (22.3)\n\tExit rate = %0.8f (17.0)\n\tNew exporter relative size = %0.8f (0.5)\n",
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
    printf("Creating panel dataset from simulation with temporary TPU...\n");
  else if(verbose && reform_flag==2)
    printf("Creating panel dataset from simulation with permanent TPU ...\n");

  time_t start, stop;
  time(&start);

  //FILE * file = 0x0;
  FILE * file2 = 0x0;
  if(reform_flag==0)
    {
      //file = fopen("output/simul_no_tpu.csv","w");
      file2 = fopen("output/simul_agg_no_tpu.csv","w");
    }
  else if(reform_flag==1)
    {
      //file = fopen("output/simul_tpu_temp.csv","w");
      file2 = fopen("output/simul_agg_tpu_temp.csv","w");
    }
  else if(reform_flag==2)
    {
      //file = fopen("output/simul_tpu_perm.csv","w");
      file2 = fopen("output/simul_agg_tpu_perm.csv","w");
    }

  //fprintf(file,"s,i,y,f,tau_applied,tau_nntr,v,entrant,incumbent,exit\n");
  fprintf(file2,"s,i,y,tau_applied,tau_nntr,exports,num_exporters,exits,entries\n");
  for(int s=0; s<NS; s++)
    {
      for(int i=0; i<NI; i++)
	{
	  if(!policy_solved_flag[i])
	    {
	      for(int t=1; t<NT; t++)
		{
		  double exports = 0.0;
		  int nf = 0;
		  int exit2=0;
		  int entrant2=0;
		  
		  for(int f=0; f<NF; f++)
		    {
		      if(v_sim[s][i][f][t]>1.0e-10)
			{
			  nf += 1;
			  exports += v_sim[s][i][f][t];
			  
			  if(gsl_isinf(exports) || gsl_isnan(exports))
			    {
			      printf("Error! Inf/Nan exports!\n");
			    }
			  
			  int exit = (t<NT-1 && v_sim[s][i][f][t+1]<1.0e-10);
			  exit2 += exit;
			  
			  int entrant = (t>1 && v_sim[s][i][f][t-1]<0.0);
			  entrant2 += entrant;
			  
			  //int incumbent = 1-entrant;
			  //fprintf(file,"%d,%s,%d,FIRM%d,%0.16f,%0.16f,%0.16f,%d,%d,%d\n",
			  //	  s,industry[i],t,f,tau_applied[i][t],tau_nntr[i][t],v_sim[s][i][f][t],exit,entrant,incumbent);
			}
		    }
		  fprintf(file2,"%d,%s,%d,%0.16f,%0.16f,%0.16f,%d,%d,%d\n",
			  s,industry[i],t,tau_applied[i][t],tau_nntr[i][t],exports,nf,exit2,entrant2);
		}
	    }
	}
    }

  //fclose(file);
  fclose(file2);

  time(&stop);

  if(verbose)
    printf("Panel data construction complete in %0.0f seconds.\n",difftime(stop,start));
}

///////////////////////////////////////////////////////////////////////////////
// 5. Calibrating TPU probs
///////////////////////////////////////////////////////////////////////////////
double caldata_temp[21];
double caldata_perm[21];
int iter=0;

int load_caldata()
{
  FILE * file = fopen("../scripts/caldata.txt","r");
  if(!file)
    {
      printf("Failed to open file with calibration data!\n");
      return 1;
    }
  else
    {
      int got = 0;
      for(int t=0; t<21; t++)
	{
	  got += fscanf(file,"%lf",&(caldata_temp[t]));
	}
      for(int t=0; t<21; t++)
	{
	  got += fscanf(file,"%lf",&(caldata_perm[t]));
	}
      fclose(file);
      if(got != 42)
	{
	  printf("Failed to load calibration data!\n");
	  return 1;
	}
      else
	{
	  return 0;
	}
    }
  
}

double calc_coeffs()
{
  linebreak2();
  printf("Computing differences between actual and simulated NNTR gap coefficients...\n");

  printf("\nProbabilities:\n");
  printf("\tYear: ");
  for(int t=t_reform; t<t_wto; t++)
    {
      printf(" %d ",t+1971);
    }
  printf("\n\tTemp:");
  for(int t=t_reform; t<t_wto; t++)
    {
      printf(" %0.3f",tpu_prob_temp[t]);
    }
  printf("\n\tPerm:");
  for(int t=t_reform; t<t_wto; t++)
    {
      printf(" %0.3f",tpu_prob_perm[t]);
    }
  printf("\n");
  
  time_t start, stop;
  time(&start);

  if(solve_policies2_tpu())
    return -99; 
  
  double expart_rate, exit_rate, new_size;
  
  simul2(1,&expart_rate,&exit_rate,&new_size,0);
  create_panel_dataset(1);

  simul2(2,&expart_rate,&exit_rate,&new_size,0);
  create_panel_dataset(2);

  time_t start2, stop2;
  time(&start2);
  printf("\nProcessing simulated data...\n");
  if(system("python3 -W ignore ../scripts/proc_simul.py"))
    return -99;
  time(&stop2);
  printf("\nProcessing complete in %0.0f seconds\n",difftime(stop2,start2));  
  
  if(load_caldata())
    return -99;

  //printf("Coeff errors:\n");
  double maxdiff = -HUGE_VAL;
  for(int t=t_reform; t<t_wto; t++)
    {
      if(fabs(caldata_temp[t-t_reform])>maxdiff)
	maxdiff = fabs(caldata_temp[t-t_reform]);

      if(fabs(caldata_perm[t-t_reform])>maxdiff)
	maxdiff = fabs(caldata_perm[t-t_reform]);

      //printf("\t%0.6f %0.6f\n",caldata_temp[t-t_reform],caldata_perm[t-t_reform]);
    }

  printf("\nErrors:\n");
  printf("\tYear: ");
  for(int t=t_reform; t<t_wto; t++)
    {
      printf(" %d ",t+1972);
    }
  printf("\n\tTemp:");
  for(int t=t_reform; t<t_wto; t++)
    {
      printf(" %+0.2f",caldata_temp[t-t_reform]);
    }
  printf("\n\tPerm:");
  for(int t=t_reform; t<t_wto; t++)
    {
      printf(" %+0.2f",caldata_perm[t-t_reform]);
    }
  printf("\n");


  time(&stop);
  printf("\nIteration %d complete in %0.0f seconds. Max error = %0.6f\n",iter,difftime(stop,start),maxdiff);

  return maxdiff;
}

void update_probs()
{
  for(int t=t_reform; t<t_wto; t++)
    {
      double err = caldata_temp[t-t_reform];
      double err2=0;
      double err3=0;
      double err4=0;
      if(t<t_wto-1)
	{
	  err2 = caldata_temp[t+1-t_reform];
	}
      if(t<t_wto-2)
	{
	  err3 = caldata_temp[t+2-t_reform];
	}
      if(t<t_wto-3)
	{
	  err3 = caldata_temp[t+3-t_reform];
	}
	
      err = err + 0.25*err2 + 0.1*err3 + 0.05*err4;
      double aerr = fabs(err);
      
      if(err>0.0)
	{
	  tpu_prob_temp[t] = tpu_prob_temp[t] * (1.0 + log(1.0+aerr) * tpu_prob_update_speed);
	}
      else
	{
	  tpu_prob_temp[t] = tpu_prob_temp[t] * (1.0 - log(1.0+aerr) * tpu_prob_update_speed);
	}

      err = caldata_perm[t-t_reform];
      aerr = fabs(err);
      if(err>0.0)
	{
	  tpu_prob_perm[t] = tpu_prob_perm[t] * (1.0 + log(1.0+aerr) * tpu_prob_update_speed);
	}
      else
	{
	  tpu_prob_perm[t] = tpu_prob_perm[t] * (1.0 - log(1.0+aerr) * tpu_prob_update_speed);
	}	
    }
  
  FILE * file = fopen("output/tpuprobs.txt","w");
  for(int t=t_reform; t<t_wto; t++)
    {
      fprintf(file,"%0.16f",tpu_prob_temp[t]);
      if(t<t_wto-1)
	fprintf(file," ");
      else
	fprintf(file,"\n");
    }
  for(int t=t_reform; t<t_wto; t++)
    {
      fprintf(file,"%0.16f",tpu_prob_perm[t]);
      if(t<t_wto-1)
	fprintf(file," ");
      else
	fprintf(file,"\n");
    }
  fclose(file);
}

int calibrate_probs()
{
  printf("Calibrating TPU probabilities to match estimated NNTR gap coefficients...\n");
  
  time_t start, stop;
  time(&start);

  //int iter=0;
  double maxdiff = +HUGE_VAL;
  do
    {
      iter++;
      maxdiff = calc_coeffs();

      if(maxdiff<0)
	{
	  printf("Error while computing coefficients! Exiting...\n");
	  return 1;
	}
      
      //printf("\nIter = %d. Max error = %0.6f\n",iter,maxdiff);
      if(maxdiff<coeff_err_tol)
	break;

      update_probs();
    }
  while(iter < max_cal_iter);

  time(&stop);
  printf("\nCalibration complete in %0.0f seconds\n",difftime(stop,start));

  return 0;
}


///////////////////////////////////////////////////////////////////////////////
// 6. Main function and setup/cleanup
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

int det_analysis()
{
  printf("Solving and simulating deterministic model...\n");

  time_t start, stop;
  time(&start);

  if(solve_policies2())
    return 1;

  //if(solve_policies2_tpu())
  //  return 1; 

  double expart_rate=0.0;
  double exit_rate=0.0;
  double new_size;
  simul2(0,&expart_rate,&exit_rate,&new_size,1);
  create_panel_dataset(0);

  //simul2(1,&expart_rate,&exit_rate,&new_size,0);
  //create_panel_dataset(1);

  //simul2(2,&expart_rate,&exit_rate,&new_size,0);
  //create_panel_dataset(2);

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
  if(det_analysis())
      return 1;

  // calibrate TPU probs
  linebreak();	  
  if(calibrate_probs())
      return 1;

  
  // finish program
  linebreak();  
  time(&stop);
  printf("\nProgram complete! Total runtime: %0.16f seconds.\n",difftime(stop,start));

  return 0;
}


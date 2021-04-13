/////////////////////////////////////////////////////////////////////////////
// 1. Includes, macros, etc.
/////////////////////////////////////////////////////////////////////////////

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
#define NI 1742 // number of industries
#define NZ 201 // productivity shock grid size
#define NT 100 // simulation length
#define NS 200 // number of simulations
#define NF 100000 // simulation population size

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
const int t_data_max = 37; // = 2008 - 1971
const int root_max_iter = 100;
const double root_tol_rel = 1.0e-6;
const double root_tol_abs = 1.0e-6;
const double delta_deriv = 1.0e-9;
const double delta_root = 1.0e-9;
const double policy_tol_abs = 1.0e-10;
const int policy_max_iter = 20000;
const double x_grid_ub_mult = 10.0;
const double x_grid_exp = 1.0;
const double tpu_prob_update_speed = 0.01;
const double coeff_err_tol = 1.0e-3;
const int max_cal_iter = 150;

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

/////////////////////////////////////////////////////////////////////////////
// 2. Declarations of parameters, grids, and inline functions
/////////////////////////////////////////////////////////////////////////////

// parameters
double W = 0.0; // wage (note: represents normalization of export country GDP per capita relative to representative destination)
double Q = 0.0; // discount factor
double delta0 = 0.0; // survival rate parameter 1
double delta1 = 0.0; // survival rate parameter 2
double delta[NZ] = {0.0}; // survival rate vector
double theta = 0.0; // EoS between varieties
double theta_hat = 0.0; // = (1/theta)*(theta/(theta-1))^(1-theta)
double theta_hat2 = 0.0; // = theta*(1/theta)*(theta/(theta-1))^(1-theta)
double sig_z = 0.0; // stochastic productivity dispersion
double rho_z = 0.0; // stochastic productivity persistence
double mu_e = 0.0; // new entrant productivity
double kmult = 0.0; // entry cost
double kappa0 = 0.0; // entry cost
double kappa1 = 0.0; // continuation cost
double xi = 0.0; // iceberg cost in high state
double rho0 = 0.0; // transition probability from low state
double rho1 = 0.0; // transition probability from high state

// productivity shock grid
double z_grid[NZ] = {0.0}; // grid
double z_hat[NZ] = {0.0}; // z^{theta-1} grid
double z_ucond_probs[NZ] = {0.0}; // ergodic probabilities
double z_ucond_cumprobs[NZ] = {0.0}; // cumulative ergodic probabilities
double z_trans_probs[NZ][NZ] = {{0.0}}; // transition probabilities
double z_trans_cumprobs[NZ][NZ] = {{0.0}}; // cumultative transition probabilities

// tariffs
char industry[NI][128] = {{""}};
double tau_applied[NI][NT] = {{0.0}}; // trade cost before liberaliztion
double tau_nntr[NI] = {0.0}; // trade cost before liberaliztion
double gap[NI] = {0.0}; // nntr_gap
double tpu_trans_mat[3][3][NT] = {{{0.0}}};

// discretization of productivity shock process
void discretize_z()
{
  int n = NZ;
  double inprob = 1.0e-8;
  double lo = gsl_cdf_ugaussian_Pinv(inprob)*sig_z*1.5;
  double hi = -gsl_cdf_ugaussian_Pinv(inprob)*sig_z*1.5;
  double ucond_std = sqrt(sig_z*sig_z/(1.0-rho_z*rho_z));
  double d = (hi-lo)/(n-1.0);
  linspace(lo,hi,n,z_grid);
  
  for(int iz=0; iz<n; iz++)
    {
      double x = z_grid[iz];

      double sum=0.0;
      for(int izp=0; izp<n; izp++)
	{
	  double y = z_grid[izp];
	  
	  z_trans_probs[iz][izp] = (gsl_cdf_ugaussian_P( (y + d/2.0 - rho_z*x) / sig_z ) -
				       gsl_cdf_ugaussian_P( (y - d/2.0 - rho_z*x) / sig_z ));
	  sum += z_trans_probs[iz][izp];
	}
      for(int izp=0; izp<n; izp++)
	{
	  z_trans_probs[iz][izp] = z_trans_probs[iz][izp]/sum;
	}
    }

  double sum=0.0;
  for(int iz=0; iz<n; iz++)
    {
      double x = z_grid[iz];
      
      z_ucond_probs[iz] = (gsl_cdf_ugaussian_P( (x + mu_e + d/2.0) / ucond_std ) -
			  gsl_cdf_ugaussian_P( (x + mu_e - d/2.0) / ucond_std ));
      sum += z_ucond_probs[iz];
    }
  for(int iz=0; iz<n; iz++)
    {
      z_ucond_probs[iz] = z_ucond_probs[iz]/sum;
    }

  sum=0.0;
  for(int iz=0; iz<n; iz++)
    {
      z_grid[iz] = exp(z_grid[iz]);
      z_hat[iz] = z_grid[iz];
      sum += z_ucond_probs[iz];
      z_ucond_cumprobs[iz] = sum;

      double sum2=0.0;
      for(int izp=0; izp<n; izp++)
	{
	  sum2 += z_trans_probs[iz][izp];
	  z_trans_cumprobs[iz][izp] = sum2;
	}
    }
}

// survival probability vector
void calc_death_probs()
{
  for(int iz=0; iz<NZ; iz++)
    {
      double death_prob=fmax(0.0,fmin(exp(-delta0*z_hat[iz])+delta1,1.0));
      delta[iz] = 1.0-death_prob;
    }
}

// assigned parameters and initial guesses
int init_params()
{
  // params constant to all industries
  W = 1.0;
  Q = 0.96;
  theta = 3.55;
  theta_hat = (1.0/3.0) * pow(3.0/(3.0-1.0),1.0-3.0);
  theta_hat2 = 3.0 * (1.0/3.0) * pow(3.0/(3.0-1.0),1.0-3.0);

  delta0 = 21.04284098;
  delta1 = 0.02258301;
  sig_z = 1.32;
  rho_z =  0.65;
  kmult = 0.0;
  
  // theta=3
  kappa0 = 0.6;
  kappa1 = 0.32;
  xi = 2.4;
            
  // mu_e[i] = 1.34;
  mu_e = 0.0;
  rho0=0.85;
  rho1=0.85;
  
  // load tariff data
  FILE * file = fopen("../scripts/path_simulation.csv","r");
  if(!file)
    {
      printf("Failed to open file with tariff data!\n");
      return 1;
    }
  else
    {
      char buffer[128];
      int t;
      double tau_, nntr_, gap_;
      int i;

      int cnt=0;
      
      while(fscanf(file,"%d %s %d %lf %lf %lf",&i,buffer,&t,&tau_,&nntr_,&gap_) == 6)
      {
	cnt+=1;

	if(gsl_isnan(tau_) || gsl_isnan(nntr_) || gsl_isnan(gap_) ||
	   gsl_isinf(tau_) || gsl_isinf(nntr_) || gsl_isinf(gap_))
	  {
	    printf("NaN or Inf detected in tariff data!\n");
	    fclose(file);
	    return 1;
	  }
	
	tau_applied[i][t-1971] = 1.0 + tau_;
	if(t==1974)
	  {
	    tau_nntr[i] = 1.0 + nntr_;
	    gap[i] = gap_;
	    strncpy(industry[i],buffer,128);
	  }
      }
      if(!feof(file))
	{
	  printf("Failed to load data!\n");
	  fclose(file);
	  return 1;
	}
      fclose(file);

      if(cnt != (NI*(t_data_max-3+1)))
	{
	  printf("Failed to load data cnt = %d, should be %d!\n",cnt,NI*(t_data_max-3));
	  return 1;
	}

      for(int i=0; i<NI; i++)
	{
	  for(int t=0; t<3; t++)
	    {
	      tau_applied[i][t] = tau_applied[i][3];
	    }
	}
      
      for(int t=t_data_max+1; t<NT; t++)
	{
	  for(int i=0; i<NI; i++)
	    {
	      tau_applied[i][t] = tau_applied[i][t-1];
	    }
	}

      for(int t=0; t<NT; t++)
	{
	  for(int i=0; i<NI; i++)
	    {
	      double tau_ = tau_applied[i][t];
	      double nntr_ = tau_nntr[i];
	      if(gsl_isnan(tau_) || gsl_isnan(nntr_) || gsl_isnan(gap_) ||
		 gsl_isinf(tau_) || gsl_isinf(nntr_) || gsl_isinf(gap_) ||
		 tau_<0.999 || nntr_ < 0.99)
		
		{
		  printf("Bad tariff data!\n");
		  printf("%d %d %0.16f %0.16f\n",t,i,tau_,nntr_);
		  return 1;
		}
	    }
	}
    }

  // 0: NNTR
  // 1: MFN
  for(int t=0; t<NT; t++)
    {
      double frac = (double)(t-t_reform)/ ((double)(t_data_max-t_reform));
      
      tpu_trans_mat[0][0][t] = 0.95;
      tpu_trans_mat[0][1][t] = 0.05;

      if(t<t_reform)
	tpu_trans_mat[1][0][t] = 0.8;
      else if(t<=t_data_max)
	tpu_trans_mat[1][0][t] = 0.8 * (1.0-sqrt(frac)) + sqrt(frac) * 0.01;
      else
	tpu_trans_mat[1][0][t] = tpu_trans_mat[1][0][t-1];
      
      tpu_trans_mat[1][1][t] = 1.0-tpu_trans_mat[1][0][t];
      }

  file = fopen("output/tpuprobs_markov.txt","r");
  if(!file)
    {
      printf("Failed to open file with probabilities!\n");
      return 1;
    }
  else
    {
      double prob;
      int cnt=0;
      for(int t=3; t<t_reform; t++)
	{
	  cnt += fscanf(file,"%lf",&prob);
	  tpu_trans_mat[0][1][t] = prob;
	}
      for(int t=t_reform; t<t_data_max; t++)
	{
	  cnt += fscanf(file,"%lf",&prob);
	  tpu_trans_mat[1][0][t] = prob;
	}
      fclose(file);

      if(cnt != t_data_max-3)
	{
	  printf("Failed to load probabilities!\n");
	  return 1;
	}

      //tpu_trans_mat[0][1][3] = 0.15;
      for(int t=0; t<3; t++)
	{
	  tpu_trans_mat[0][1][t] = tpu_trans_mat[0][1][3];
	}
      for(int t=t_reform; t<NT; t++)
	{
	  tpu_trans_mat[0][1][t] = tpu_trans_mat[0][1][3];
	}
      
      for(int t=0; t<t_reform; t++)
	{
	  tpu_trans_mat[1][0][t] = tpu_trans_mat[1][0][t_reform];
	}
      for(int t=t_data_max; t<NT; t++)
	{
	  tpu_trans_mat[1][0][t] = tpu_trans_mat[1][0][t_data_max-1];
	}

      for(int t=0; t<NT; t++)
	{
	  tpu_trans_mat[0][0][t] = 1.0-tpu_trans_mat[0][1][t];
	  tpu_trans_mat[1][1][t] = 1.0-tpu_trans_mat[1][0][t];
	}
    }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////
// 3. Dynamic program: deterministic
/////////////////////////////////////////////////////////////////////////////

double V_nntr[NI][NZ][3] = {{{0.0}}}; // NNTR
double V_ref_det[NI][NZ][3][NT] = {{{{0.0}}}}; // perfect foresight
double EV_det[NI][NZ][3] = {{{0.0}}}; // continuation value storage

int gex_nntr[NI][NZ][3] = {{{0}}}; // NNTR
int gex_ref_det[NI][NZ][3][NT] = {{{{0}}}}; // perfect foresight

int policy_solved_flag[NI] = {0};

// initial guess for value functions
void init_dp_objs_det()
{
  for(int i=0; i<NI; i++)
    {
      for(int iz=0; iz<NZ; iz++)
	{ 
	  double pi_hat = theta_hat * pow(tau_nntr[i],-theta);
	  V_nntr[i][iz][0] = 0.0;
	  V_nntr[i][iz][1] = pi_hat*pow(xi,1.0-theta)*z_hat[iz]/Q;
	  V_nntr[i][iz][2] = pi_hat*z_hat[iz]/Q;
	  
	  for(int t=0; t<NT; t++)
	    {
	      pi_hat = theta_hat * pow(tau_applied[i][t],-theta);
	      V_ref_det[i][iz][0][t] = 0.0;
	      V_ref_det[i][iz][1][t] = pi_hat*pow(xi,1.0-theta)*z_hat[iz]/Q;
	      V_ref_det[i][iz][2][t] = pi_hat*z_hat[iz]/Q;
	    }
	}
    }
}

// steady state Bellman
void calc_EV_det(int i, int t, int flag)
{
    for(int z=0; z<NZ; z++)
    {
      for(int e=0; e<3; e++)
	{
	  if(flag==0)
	    {
	      EV_det[i][z][e]=0.0;
	    }
	  else
	    {
	      EV_det[i][z][e]=0.0;
	    }
	  
	  for(int zp=0; zp<NZ; zp++)
	    {
	      if(z_trans_probs[z][zp]>1.0e-11)
		{
		  if(flag==0)
		    {
		      EV_det[i][z][e] += Q*delta[z]*
			V_nntr[i][zp][e]*z_trans_probs[z][zp];
		    }
		  else
		    {
		      EV_det[i][z][e] += Q*delta[z]*
			V_ref_det[i][zp][e][t]*z_trans_probs[z][zp];
		    }
		}
	    }
	} 
    }

}

void iterate_policies_det(int i, int t, int flag,
			  double * maxdiff, int imaxdiff[3])
{
  *maxdiff=-HUGE_VAL;

  if(t<NT-1)
    calc_EV_det(i,t+1,flag);
  else
    calc_EV_det(i,t,flag);
 
  for(int z=0; z<NZ; z++)
    {
      int * gex1, * gex2, * gex3;
      if(flag==0)
	{
	  gex1 = &(gex_nntr[i][z][0]);
	  gex2 = &(gex_nntr[i][z][1]);
	  gex3 = &(gex_nntr[i][z][2]);
	}
      else
	{
	  gex1 = &(gex_ref_det[i][z][0][t]);
	  gex2 = &(gex_ref_det[i][z][1][t]);
	  gex3 = &(gex_ref_det[i][z][2][t]);
	}
    

      // applied
      if(EV_det[i][z][0] < EV_det[i][z][1] - kappa0)
	{
	  *gex1 = 1;
	}
      else
	{
	  *gex1 = 0;
	}

      if(EV_det[i][z][0] <
	 rho0*EV_det[i][z][1] + (1.0-rho0)*EV_det[i][z][2] - kappa1)
	{
	  *gex2 = 1;
	}
      else
	{
	  *gex2 = 0;
	}
	  
      if(EV_det[i][z][0] <
	 (1.0-rho1)*EV_det[i][z][1] + rho1*EV_det[i][z][2] - kappa1)
	{
	  *gex3 = 1;
	}
      else
	{
	  *gex3 = 0;
	}
      
      // update continuation values and check convergence ---------------
      double pi=0.0;
      if(flag==0 || t<t_reform)
	pi = theta_hat * pow(tau_nntr[i],-theta) *z_hat[z];
      else
	pi = theta_hat * pow(tau_applied[i][t],-theta) *z_hat[z];
      
      double tmp0 = fmax(EV_det[i][z][0], EV_det[i][z][1] - kappa0);
      
      double tmp1 = pi*pow(xi,1.0-theta) +
	fmax(EV_det[i][z][0],
	     rho0*EV_det[i][z][1] +
	     (1.0-rho0)*EV_det[i][z][2] - kappa1);

      double tmp2 = pi +
	fmax(EV_det[i][z][0],
	     (1.0-rho1)*EV_det[i][z][1] +
	     rho1*EV_det[i][z][2] - kappa1);

	
      double diff0=0, diff1=0, diff2=0;
      if(flag==0)
	{
	  diff0 = fabs(tmp0-V_nntr[i][z][0]);
	  diff1 = fabs(tmp1-V_nntr[i][z][1]);
	  diff2 = fabs(tmp2-V_nntr[i][z][2]);
	}
      else
	{
	  diff0 = fabs(tmp0-V_ref_det[i][z][0][t]);
	  diff1 = fabs(tmp1-V_ref_det[i][z][1][t]);
	  diff2 = fabs(tmp2-V_ref_det[i][z][2][t]);
	}

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

      if(flag==0)
	{
	  V_nntr[i][z][0] = tmp0;
	  V_nntr[i][z][1] = tmp1;
	  V_nntr[i][z][2] = tmp2;
	}
      else
	{
	  V_ref_det[i][z][0][t] = tmp0;
	  V_ref_det[i][z][1][t] = tmp1;
	  V_ref_det[i][z][2][t] = tmp2;
	}
    }
}

// solve policy function for industry i
int solve_policies_det(int i)
{
  init_dp_objs_det(i);

  double maxdiff = 999;
  int imaxdiff[3];

  // first do SS policy for NNTR rates
  int iter=0;
  do
    {
      iter++;
      iterate_policies_det(i,NT-1,0,&maxdiff,imaxdiff);
    }
  while(maxdiff>policy_tol_abs && iter < policy_max_iter);

  if(iter==policy_max_iter)
    {
      printf("\tValue function iteration failed for industry %d! Diff = %0.4g\n",i,maxdiff);
      return 1;
    }

  // now do SS policy for very last period of MFN rates
  do
    {
      iter++;
      iterate_policies_det(i,NT-1,1,&maxdiff,imaxdiff);
    }
  while(maxdiff>policy_tol_abs && iter < policy_max_iter);

  if(iter==policy_max_iter)
    {
      printf("\tValue function iteration failed for industry %d! Diff = %0.4g\n",i,maxdiff);
      return 1;
    }

  // now iterate backwards to get path of MFN policies
  for(int t=NT-2; t>=0; t--)
    {
      iterate_policies_det(i,t,1,&maxdiff,imaxdiff);
    }

  return 0;
}

// solve policies for all industries in parallel
int solve_policies2_det()
{
  if(verbose)
    printf("\nSolving deterministic dynamic program...\n");

  time_t start, stop;
  time(&start);

  int cnt=0;
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<NI; i++)
    {
      policy_solved_flag[i] = solve_policies_det(i);
      cnt += policy_solved_flag[i];
    }

  time(&stop);
  
  if(verbose)
    {
      printf("Finished %0.0f seconds. %d failed to converge.\n",difftime(stop,start),cnt);
    }
  
  return 0;  
}

/////////////////////////////////////////////////////////////////////////////
// 4. Dynamic program: Markov process
/////////////////////////////////////////////////////////////////////////////

double V_markov[NI][NZ][3][2][NT] = {{{{{0.0}}}}}; // Markov process
double EV_markov[NI][NZ][3][2] = {{{{0.0}}}}; // Markov process
int gex_markov[NI][NZ][3][2][NT] = {{{{{0}}}}}; // Markov process

// initial guess for value functions
void init_dp_objs_markov()
{
  for(int i=0; i<NI; i++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  double pi_hat = theta_hat * pow(tau_applied[i][NT-1],-theta);
	  V_markov[i][iz][0][1][NT-1] = 0.0;
	  V_markov[i][iz][1][1][NT-1] = pi_hat*pow(xi,1.0-theta)*z_hat[iz]/Q;
	  V_markov[i][iz][2][1][NT-1] = pi_hat*z_hat[iz]/Q;
	  
	  pi_hat = theta_hat * pow(tau_nntr[i],-theta);
	  V_markov[i][iz][0][0][NT-1] = 0.0;
	  V_markov[i][iz][1][0][NT-1] = pi_hat*pow(xi,1.0-theta)*z_hat[iz]/Q;
	  V_markov[i][iz][2][0][NT-1] = pi_hat*z_hat[iz]/Q;
	}
    }
}

// steady state Bellman
void calc_EV_markov(int i, int t)
{
    for(int z=0; z<NZ; z++)
    {
      for(int e=0; e<3; e++)
	{
	  for(int p=0; p<2; p++)
	    {
	      EV_markov[i][z][e][p]=0.0;
	      for(int zp=0; zp<NZ; zp++)
		{
		  if(z_trans_probs[z][zp]>1.0e-11)
		    {
		      for(int pp=0; pp<2; pp++)
			{
			  EV_markov[i][z][e][p] += Q*delta[z]*
			    V_markov[i][zp][e][pp][t]*
			    z_trans_probs[z][zp]*tpu_trans_mat[p][pp][t];
			}
		    }
		}
	    }
	} 
    }

}

void iterate_policies_markov(int i, int t, double * maxdiff, int imaxdiff[3])
{
  *maxdiff=-HUGE_VAL;

  if(t==NT-1)
    calc_EV_markov(i,t);
  else
    calc_EV_markov(i,t+1);
    
  
  for(int z=0; z<NZ; z++)
    {
      // first compute policy functions -------------------
      
      // NNTR = 0, MFN = 1
      for(int p=0; p<2; p++)
	{
	  if(EV_markov[i][z][0][p] < EV_markov[i][z][1][p] - kappa0)
	    {
	      gex_markov[i][z][0][p][t] = 1;
	    }
	  else
	    {
	      gex_markov[i][z][0][p][t] = 0;
	    }

	  if(EV_markov[i][z][0][p] <
	     rho0*EV_markov[i][z][1][p] +
	     (1.0-rho0)*EV_markov[i][z][2][p] - kappa1)
	    {
	      gex_markov[i][z][1][p][t] = 1;
	    }
	  else
	    {
	      gex_markov[i][z][1][p][t] = 0;
	    }
	  
	  if(EV_markov[i][z][0][p] <
	     (1.0-rho1)*EV_markov[i][z][1][p] +
	     rho1*EV_markov[i][z][2][p] - kappa1)
	    {
	      gex_markov[i][z][2][p][t] = 1;
	    }
	  else
	    {
	      gex_markov[i][z][2][p][t] = 0;
	    }
	}
      
      // update continuation values and check convergence ---------------

      // autarky
      for(int p=0; p<2; p++)
	{
	  double tmp0=0.0;
	  double tmp1=0.0;
	  double tmp2=0.0;
	  
	  double pi = 0.0;
	  if(p==0)
	    pi = theta_hat * pow(tau_nntr[i],-theta) *z_hat[z];
	  else if(t<t_reform)
	    pi = theta_hat * pow(tau_applied[i][t_reform],-theta) *z_hat[z];
	  else
	    pi = theta_hat * pow(tau_applied[i][t],-theta) *z_hat[z];
		

	  tmp0 = fmax(EV_markov[i][z][0][p],
		      EV_markov[i][z][1][p] - kappa0);
      
	  tmp1 = pi*pow(xi,1.0-theta) +
	    fmax(EV_markov[i][z][0][p],
		 rho0*EV_markov[i][z][1][p] +
		 (1.0-rho0)*EV_markov[i][z][2][p] - kappa1);

	  tmp2 = pi +
	    fmax(EV_markov[i][z][0][p],
		 (1.0-rho1)*EV_markov[i][z][1][p] +
		 rho1*EV_markov[i][z][2][p] - kappa1);
	  
	  double diff0 = fabs(tmp0-V_markov[i][z][0][p][t]);
	  double diff1 = fabs(tmp1-V_markov[i][z][1][p][t]);
	  double diff2 = fabs(tmp2-V_markov[i][z][2][p][t]);

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

	  V_markov[i][z][0][p][t] = tmp0;
	  V_markov[i][z][1][p][t] = tmp1;
	  V_markov[i][z][2][p][t] = tmp2;
	}
    }
}

// solve policy function for industry i
int solve_policies_markov(int i)
{
  init_dp_objs_markov(i);

  double maxdiff = 999;
  int imaxdiff[3];

  // first do SS policies for pre-1980 and NNTR rates
  int iter=0;
  do
    {
      iter++;
      iterate_policies_markov(i,NT-1,&maxdiff,imaxdiff);
    }
  while(maxdiff>policy_tol_abs && iter < policy_max_iter);

  if(iter==policy_max_iter)
    {
      printf("\tValue function iteration failed for industry %d! Diff = %0.4g\n",i,maxdiff);
      return 1;
    }

  // now iterate backwards
  for(int t=NT-2; t>=0; t--)
    {
      iterate_policies_markov(i,t,&maxdiff,imaxdiff);
    }

  return 0;
}

// solve policies for all industries in parallel
int solve_policies2_markov()
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
      policy_solved_flag[i] = solve_policies_markov(i);
      cnt += policy_solved_flag[i];
    }

  time(&stop);
  
  if(verbose)
    {
      printf("Finished dynamic programs in %0.0f seconds. %d failed to converge.\n",difftime(stop,start),cnt);
    }
  
  return 0;  
}


/////////////////////////////////////////////////////////////////////////////
// 5. Simulation
/////////////////////////////////////////////////////////////////////////////

// storage for simulated data
// we use first NT periods to simulate convergence to pre-reform steady state and NT periods for post-reform dynamics
unsigned long int seed = 0;
double i_rand[NS][NF];
double z_rand[NS][NF][NT];
double switch_rand[NS][NF][NT];
double surv_rand[NS][NF][NT];
int i_sim[NS][NF];
double v_sim[NS][NF][NT];

// draw random variables
void random_draws()
{
  printf("\nDrawing random numbers for simulation...\n");
  
  time_t start, stop;
  time(&start);
  
  gsl_rng_env_setup();
  gsl_rng * r = gsl_rng_alloc(gsl_rng_default);

  for(int f=0; f<NF; f++)
    {	
      for(int s=0; s<NS; s++)
	{
	  i_rand[s][f] = gsl_rng_uniform(r);
	  
	  for(int t=0; t<NT; t++)
	    {
	      z_rand[s][f][t] = gsl_rng_uniform(r);
	      switch_rand[s][f][t] = gsl_rng_uniform(r);
	      surv_rand[s][f][t] = gsl_rng_uniform(r);
	    }
	}
    }

  gsl_rng_free(r);

  gsl_interp_accel * acc = gsl_interp_accel_alloc();
  double iprobs[NI] = {0.0};
  for(int i=0; i<NI; i++)
    {
      iprobs[i] = 1.0/((double)(NI));
    }

    for(int s=0; s<NS; s++)
    {
      for(int f=0; f<NF; f++)
	{
	  i_sim[s][f] = gsl_interp_accel_find(acc, iprobs, NI, i_rand[s][f]);
	}
    }
  gsl_interp_accel_free(acc);


  time(&stop);
  printf("Random draws finished! Time = %0.0f\n",difftime(stop,start));
}

// main simulation function
void simul(int s, int reform_flag)
{
  time_t start, stop;
  time(&start);

  gsl_interp_accel * acc1 = gsl_interp_accel_alloc();

  // for each firm in the sample...
  for(int f=0; f<NF; f++)
    {      
      // find initial value of shock based on random draw and ergodic distribution	
      gsl_interp_accel_reset(acc1);
      int z = gsl_interp_accel_find(acc1, z_ucond_cumprobs, NZ, z_rand[s][f][0]);
      int i = i_sim[s][f];
      
      // start off as a non-exporter
      int e=0;

      // loop over the time periods in the simulation
      for(int t=0; t<NT; t++)
	{
	  // determine which profit multiplier to use depending on the time period
	  double tau_hat_ = pow(tau_applied[i][t],-theta);

	  if(t<t_reform || reform_flag==2)
	    tau_hat_ = pow(tau_nntr[i],-theta);
	  else if(t==t_reform && reform_flag != 2)
	    tau_hat_ = 0.5*pow(tau_applied[i][t],-theta) + 0.5*pow(tau_nntr[i],-theta);
	  
	  if(e==0) // if it is currently a non-exporter
	    {
	      v_sim[s][f][t] = -99.9;
	    }
	  else if (e==1) // if it is currently a bad exporter...
	    {
	      v_sim[s][f][t] = theta_hat2*tau_hat_*z_hat[z]*pow(xi,1.0-theta); // compute exports
	    }
	  else if(e==2) // if it is current a good exporter...
	    {
	      v_sim[s][f][t] = theta_hat2*tau_hat_*z_hat[z]; // compute exports
	    }

	  if(gsl_isinf(v_sim[s][f][t]) || gsl_isnan(v_sim[s][f][t]))
	    {
	      printf("Error! Inf/Nan exports!\n");
	      printf("tau_hat = %0.6f\n",tau_hat_);
	      printf("tau_applied = %0.6f\n",tau_applied[i][t]);
	      printf("tau_nntr = %0.6f\n",tau_nntr[i]);
	      return;
	    }

	  // determine which policy function to use depending on the time period and reform flag
	  int gex_=0;
	  
	    /*if(t>=t_wto)
	    {
	      if(reform_flag == 2)
		gex_ = gex_nntr[i][z][e];
	      else if(reform_flag<3)
		gex_ = gex_ref_det[i][z][e][t];
	      else
		gex_ = gex_markov[i][z][e][1][t];
		}*/
	  if(t<t_reform) // if we are before the 1980 reform, use the pre-MIT shock policy
	    {
	      if(reform_flag==0)
		gex_ = gex_ref_det[i][z][e][t];
	      else if(reform_flag<3)
		gex_ = gex_nntr[i][z][e];
	      else
		gex_ = gex_markov[i][z][e][0][t];
	    }
	  else
	    {// otherwise it depends on which scenarion we are using
	      if(reform_flag==0 || reform_flag==1) // no TPU
		{
		  if(t>t_data_max+10)
		    {
		      gex_ = gex_ref_det[i][z][e][t_data_max+10];
		    }
		  else
		    {
		      gex_ = gex_ref_det[i][z][e][t];
		    }
		}
	      else if(reform_flag == 2)
		{
		  gex_ = gex_nntr[i][z][e];
		}
	      else if(reform_flag==3)
		{
		  gex_ = gex_markov[i][z][e][1][t];
		}
	    }

	  // if the firm dies, exit and draw a new productivity
	  if(surv_rand[s][f][t]>delta[z])
	    {
	      e=0;
	      
	      if(t<NT-1)
		z = gsl_interp_accel_find(acc1, z_ucond_cumprobs,
					  NZ, z_rand[s][f][t+1]);
	    }
	  else
	    {
	      if(t<NT-1)
		z = gsl_interp_accel_find(acc1, z_trans_cumprobs[z],
					  NZ, z_rand[s][f][t+1]);
	      
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
			  if(switch_rand[s][f][t]<rho0)
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
			  if(switch_rand[s][f][t]<rho1)
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
    printf("\tSimulation %d completed in %0.0f seconds.\n",s,difftime(stop,start));

  return;
}

// do all simulations in parallel
void simul2(int reform_flag,
	    double *avg_expart_rate,
	    double *avg_exit_rate,
	    double *avg_new_size,
	    double *avg_5yr_gr,
	    int calc_moments)
{
  if(verbose && reform_flag==0)
    printf("\nDeterministic simulation where entire path of tariffs is anticipated...\n");
  else if(verbose && reform_flag==1)
    printf("\nDeterministic simulation where 1980 reform is unanticipated...\n");
  else if(verbose && reform_flag==2)
    printf("\nDeterministic simulation where 1980 reform never occurs...\n");
  else if(verbose && reform_flag==3)
    printf("\nSimulation with Markov process for TPU...\n");

  time_t start, stop;
  time(&start);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int s=0; s<NS; s++)
    {
      {
	simul(s,reform_flag);
      }
    }

  if(calc_moments)
    {
      // compute average export participation rate, exit rate, and new exporter size across all simulations in steady state
      int t=NT-2;
      *avg_expart_rate=0.0;
      *avg_exit_rate=0.0;
      *avg_new_size=0.0;
      *avg_5yr_gr=0.0;
      int num_sims_exit=0;
      int num_sims_newsize=0;
      int num_sims_5yr=0;
  
      for(int s=0; s<NS; s++)
	{
	  int num_exporters=0;
	  int num_firms=0;
	  int num_exits=0;
	  int num_new_exporters=0;
	  //int num_incumbents=0;
	  int num_5yr = 0.0;
	  
	  double newsize=0.0;
	  double meansize=0.0;
	  double mean5yr=0.0;
	  double mean1yr=0.0;
      
	  num_firms += NF;
	  for(int f=0; f<NF; f++)
	    {
	      if(v_sim[s][f][t]>1.0e-10)
		{
		  num_exporters += 1;
		  meansize += v_sim[s][f][t];
		  
		  if(v_sim[s][f][t+1]<0.0)
		    {
		      num_exits += 1;
		    }
		  
		  if(v_sim[s][f][t-1]<0.0)
		    {
		      num_new_exporters += 1;
		      newsize += v_sim[s][f][t];
		    }
		  
		  if(t>=5
		     && v_sim[s][f][t-1]>1.0e-10
		     && v_sim[s][f][t-2]>1.0e-10
		     && v_sim[s][f][t-3]>1.0e-10
		     && v_sim[s][f][t-4]>1.0e-10
		     && v_sim[s][f][t-5]<0.0)
		    {
		      num_5yr +=1;
		      mean5yr += v_sim[s][f][t];
		      mean1yr += v_sim[s][f][t-4];
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

	  if(num_5yr>0)
	    {
	      *avg_5yr_gr  += (mean5yr/mean1yr);
	      num_sims_5yr += 1;
	    }
	}
  
      *avg_expart_rate = *avg_expart_rate/((double)(NS));
      *avg_exit_rate = *avg_exit_rate/((double)(num_sims_exit));
      *avg_new_size  = *avg_new_size/((double)(num_sims_newsize));
      *avg_5yr_gr  = *avg_5yr_gr/((double)(num_sims_5yr));
    }

  time(&stop);

  if(verbose)
    {
      printf("Finished simulations in %0.0f seconds.\n",difftime(stop,start));
      if(calc_moments)
	{
	  printf("\tExport part. rate = %0.8f (22.3)\n\tExit rate = %0.8f (17.0)\n\tNew exporter relative size = %0.8f (0.5)\n\tAvg. 5-yr growth = %0.8f (2.0)\n",
		 100*(*avg_expart_rate),100*(*avg_exit_rate),*avg_new_size,*avg_5yr_gr);
	}
    }
    
  return;
}

// create panel dataset from simulation
void create_panel_dataset(int reform_flag)
{
  printf("Creating panel dataset from simulation %d...\n",reform_flag);

  time_t start, stop;
  time(&start);

   FILE * file2 = 0x0;
  if(reform_flag==0)
    {
      file2 = fopen("output/simul_agg_det0.csv","w");
    }
  else if(reform_flag==1)
    {
      file2 = fopen("output/simul_agg_det1.csv","w");
    }
  else if(reform_flag==2)
    {
      file2 = fopen("output/simul_agg_det2.csv","w");
    }
  else if(reform_flag==3)
    {
      file2 = fopen("output/simul_agg_tpu_markov.csv","w");
    }

  fprintf(file2,"i,y,tau_applied,tau_nntr,gap,exports,num_exporters,exits,entries\n");
  for(int t=1; t<NT; t++)
    {
      double exports[NI] = {0.0};
      int nf[NI] = {0};
      int exit2[NI] = {0};
      int entrant2[NI] ={0};

      for(int s=0; s<NS; s++)
	{
	  for(int f=0; f<NF; f++)
	    {
	      int i = i_sim[s][f];
	      
	      if(v_sim[s][f][t]>1.0e-10)
		{
		  nf[i] += 1;
		  exports[i] += v_sim[s][f][t];
		  
		  if(gsl_isinf(exports[i]) || gsl_isnan(exports[i]))
		    {
		      printf("Error! Inf/Nan exports!\n");
		    }
		  
		  int exit = (t<NT-1 && v_sim[s][f][t+1]<1.0e-10);
		  exit2[i] += exit;
		  
		  int entrant = (t>1 && v_sim[s][f][t-1]<0.0);
		  entrant2[i] += entrant;
		}
	    }
	}

      
      for(int i=0; i<NI; i++)
	{
	  exports[i] = exports[i]/NS;
	  nf[i] = nf[i]/NS;
	  exit2[i] = exit2[i]/NS;
	  entrant2[i] = entrant2[i]/NS;
	  
	  fprintf(file2,"%s,%d,%0.16f,%0.16f,%0.16f,%0.16f,%d,%d,%d\n",
		  industry[i],t,tau_applied[i][t],tau_nntr[i],gap[i],exports[i],nf[i],exit2[i],entrant2[i]);
	}
    }

  fclose(file2);

  time(&stop);

  if(verbose)
    printf("Panel data construction complete in %0.0f seconds.\n",difftime(stop,start));
}

/////////////////////////////////////////////////////////////////////////////
// 5. Trade dynamics via distribution iteration
/////////////////////////////////////////////////////////////////////////////

const double dist_tol = 1.0e-11;
const int max_dist_iter = 5000;

double dist[NI][NZ][3] = {{{0.0}}};
double tmp_dist[NI][NZ][3] = {{{0.0}}};

double exports[NI][NT];
double nf[NI][NT];

// initialize distribution
void init_dist()
{
  for(int i=0; i<NI; i++)
    {
      double sum=0.0;
      for(int iz=0; iz<NZ; iz++)
	{
	  tmp_dist[i][iz][0] = 0.0;
	  tmp_dist[i][iz][1] = 0.0;
	  tmp_dist[i][iz][2] = 0.0;
	  
	  dist[i][iz][0] = z_ucond_probs[iz];
	  dist[i][iz][1] = 0.0;
	  dist[i][iz][2] = 0.0;
	  sum += dist[i][iz][0];
	}
      if(fabs(sum-1.0)>1.0e-8)
	{
	  printf("\nInitial distribution does not sum to one! i = %d, sum = %0.4g\n",i,sum);
	}

    }
}

// distribution iteration driver
int update_dist(int i, int t, int reform_flag)
{
  for(int iz=0; iz<NZ; iz++)
    {
      for(int is=0; is<3; is++)
	{
	  tmp_dist[i][iz][is]=0.0;
	}
    }

  for(int iz=0; iz<NZ; iz++)
    {
      double surv_prob = delta[iz];
      
      for(int is=0; is<3; is++)
	{ 
	  int gex_ = 0;

	  // if we are before the 1980 reform, use the
	  // pre-MIT shock policy
	  if(t<t_reform)
	    {
	      if(reform_flag==0)
		gex_ = gex_ref_det[i][iz][is][t];
	      else if(reform_flag<3)
		gex_ = gex_nntr[i][iz][is];
	      else
		gex_ = gex_markov[i][iz][is][0][t];
	    }
	  
	  // otherwise it depends on which scenario we are using
	  else
	    {
	      if(reform_flag==0 || reform_flag==1)
		{
		  /*
		  if(t>t_data_max+10)
		    {
		      gex_ = gex_ref_det[i][iz][is][t_data_max+10];
		    }
		  else
		  {*/
		  gex_ = gex_ref_det[i][iz][is][t];
		    //}
		}
	      else if(reform_flag == 2)
		{
		  gex_ = gex_nntr[i][iz][is];
		}
	      else if(reform_flag==3)
		{
		  gex_ = gex_markov[i][iz][is][1][t];
		}
	    }

	  for(int izp=0; izp<NZ; izp++)
	    {
	      tmp_dist[i][izp][0] += (1.0-surv_prob)*
		dist[i][iz][is]*z_ucond_probs[izp];

	      if(gex_==1)
		{
		  if(is==0)
		    {
		      tmp_dist[i][izp][1] += dist[i][iz][is]*
			surv_prob*z_trans_probs[iz][izp];
		    }
		  else if(is==1)
		    {
		      tmp_dist[i][izp][1] += dist[i][iz][is]*
			surv_prob*z_trans_probs[iz][izp]*rho0;
		      
		      tmp_dist[i][izp][2] += dist[i][iz][is]*
			surv_prob*z_trans_probs[iz][izp]*(1.0-rho0);
		    }
		  else if(is==2)
		    {
		      tmp_dist[i][izp][1] += dist[i][iz][is]*
			surv_prob*z_trans_probs[iz][izp]*(1.0-rho1);
		      
		      tmp_dist[i][izp][2] += dist[i][iz][is]*
			surv_prob*z_trans_probs[iz][izp]*rho1;
		    }
		}
	      else
		{
		  tmp_dist[i][izp][0] += surv_prob*
		    dist[i][iz][is]*z_trans_probs[iz][izp];
		}
	      
	    }
	}
    }

  double sum = 0.0;
  for(int iz=0; iz<NZ; iz++)
    {
      for(int is=0; is<3; is++)
	{
	  sum = sum+tmp_dist[i][iz][is];
	}
    }

  if(fabs(sum-1.0)>1.0e-8)
    {
      printf("\nUpdated distribution does not sum to one! i = %d, t = %d, sum = %0.16f\n",i,t,sum);
      return 1;
    }

  return 0;
}

void calc_trans_vars(int i, int t, int reform_flag)
{ 
  double expart_rate=0.0;
  double total_exports=0.0;
  
  double tau_hat_ = pow(tau_applied[i][t],-theta);
  
  if(t<t_reform || reform_flag==2)
    tau_hat_ = pow(tau_nntr[i],-theta);
  else if(t==t_reform && reform_flag != 2)
    tau_hat_ = 0.5*pow(tau_applied[i][t],-theta) +
      0.5*pow(tau_nntr[i],-theta);
  
  for(int z=0; z<NZ; z++)
    {
      for(int is=0; is<3; is++)
	{
	  double d=dist[i][z][is];
	  double v=0;
	  int s=0;
	  if (is==1) // if it is currently a bad exporter...
	    {
	      v += theta_hat2*tau_hat_*
		z_hat[z]*pow(xi,1.0-theta);
	      s=1;
	    }
	  else if(is==2) // if it is current a good exporter...
	    {
	      v += theta_hat2*tau_hat_*z_hat[z];
	      s=1;
	    }

	  total_exports += v * d;
	  expart_rate += s*d;
	}
    }

  exports[i][t] = total_exports;
  nf[i][t] = expart_rate;
  
  return;
}

void do_trans_dyn(int reform_flag)
{
  printf("\nComputing transition dynamics for scenario %d...\n",reform_flag);

  time_t start, stop;
  time(&start);

  init_dist();
  
  for(int t = 0; t<NT; t++)
    {
      
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<NI; i++)
	{
	  calc_trans_vars(i,t, reform_flag);
	  update_dist(i,t, reform_flag);
	}
      
      memcpy(dist,tmp_dist,sizeof(double)*(NI*NZ*3));
    }

  FILE * file2 = 0x0;
  if(reform_flag==0)
    {
      file2 = fopen("output/simul_agg_det0.csv","w");
    }
  else if(reform_flag==1)
    {
      file2 = fopen("output/simul_agg_det1.csv","w");
    }
  else if(reform_flag==2)
    {
      file2 = fopen("output/simul_agg_det2.csv","w");
    }
  else if(reform_flag==3)
    {
      file2 = fopen("output/simul_agg_tpu_markov.csv","w");
    }

  fprintf(file2,"i,y,tau_applied,tau_nntr,gap,exports,num_exporters\n");

  for(int i=0; i<NI; i++)
    {
      for(int t=1; t<NT; t++)
	{
	  fprintf(file2,"%s,%d,%0.16f,%0.16f,%0.16f,%0.16f,%0.16f\n",
		  industry[i],t,tau_applied[i][t],tau_nntr[i],gap[i],exports[i][t],nf[i][t]);
	}
    }
  
  fclose(file2);

  time(&stop);

  if(verbose)
    printf("Transitions complete in %0.0f seconds.\n",difftime(stop,start));

}

void calc_cal_moments(double *expart_rate,
		      double *exit_rate,
		      double *new_size)
{

  *expart_rate = 0;
  *exit_rate = 0;
  *new_size = 0;

  double sum=0;
  double avg_size =0;
  double new_mass = 0;
  
  for(int z=0; z<NZ; z++)
    {
      double Ezh = 0;
      for(int zp=0; zp<NZ; zp++)
	{
	  Ezh += z_trans_probs[z][zp] * z_hat[zp];
	}
      	  
      for(int i=0; i<NI; i++)
	{
	  double tau_hat_ = pow(tau_applied[i][NT-1],-theta);
	  
	  for(int e=0; e<3; e++)
	    {
	      double d = dist[i][z][e];
	      sum += d;

	      if(e>0)
		{
		  *expart_rate += d;

		  if(gex_ref_det[i][z][e][NT-1]==0)
		    *exit_rate += d;
		  else
		    *exit_rate += (1.0-delta[z])*d;

		  if(e==1)
		    avg_size += theta_hat2*tau_hat_*z_hat[z]*
		      pow(xi,1.0-theta)*d;
		  else
		    avg_size += theta_hat2*tau_hat_*z_hat[z]*d;
		}
	      
	      if(e==0 && gex_ref_det[i][z][e][NT-1]==1)
		{
		  *new_size += theta_hat2*tau_hat_*Ezh*
		    pow(xi,1.0-theta)*d*delta[z];
		  new_mass += d*delta[z];
		}
		
	    }
	}
    }

  
  *exit_rate = *exit_rate / *expart_rate;
  *new_size = (*new_size/new_mass) / (avg_size/ (*expart_rate));
  *expart_rate = *expart_rate / sum;

  printf("Export participation rate = %0.6f\n",*expart_rate);
  printf("Exit rate rate = %0.6f\n",*exit_rate);
  printf("Rel. size of new exporters = %0.6f\n",*new_size);
}



/////////////////////////////////////////////////////////////////////////////
// 5. Calibrating TPU probs
/////////////////////////////////////////////////////////////////////////////
double caldata[NT]={0.0};
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
      for(int t=3; t<t_data_max; t++)
	{
	  got += fscanf(file,"%lf",&(caldata[t]));
	}
      fclose(file);
      if(got != t_data_max-3)
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

double calc_coeffs(int reform_flag)
{
  linebreak2();
  printf("Computing differences between actual and simulated NNTR gap coefficients...\n\n");

  if(reform_flag!=3)
    {
      printf("Wrong reform code!\n");
      return -99;
      }

  printf("P(NNTR-->MFN) = %0.3f\n",tpu_trans_mat[0][1][0]);
  printf("P(MFN-->NNTR) =");
  for(int t=t_reform; t<t_data_max; t++)
    {
      printf(" %0.2f",tpu_trans_mat[1][0][t]);
    }

  FILE * file = fopen("output/tpuprobs_markov.txt","w");
  int cnt=0;
  for(int t=3; t<t_reform; t++)
    {
      cnt++;
      fprintf(file,"%0.16f ",tpu_trans_mat[0][1][t]);
    }
  for(int t=t_reform; t<t_data_max; t++)
    {
      cnt++;
      fprintf(file,"%0.16f ",tpu_trans_mat[1][0][t]);
    }
  fclose(file);
  printf("cnt = %d\n\n",cnt);
  
  time_t start, stop;
  time(&start);

  if(solve_policies2_markov())
    return -99;
  
  //double expart_rate, exit_rate, new_size, avg_5yr_gr;
  //simul2(reform_flag,&expart_rate,&exit_rate,&new_size,&avg_5yr_gr,0);
  //create_panel_dataset(reform_flag);
  do_trans_dyn(reform_flag);
  
  time_t start2, stop2;
  time(&start2);
  printf("\nProcessing simulated data...\n");
  char buffer[128];
  sprintf(buffer,"python3 -W ignore ../scripts/proc_simul.py %d",reform_flag);
  if(system(buffer))
    return -99;
  time(&stop2);
  printf("\nProcessing complete in %0.0f seconds\n",difftime(stop2,start2));  

  if(load_caldata())
    return -99;

  double retval = -HUGE_VAL;
  if(reform_flag==2)
    {
      for(int t=t_reform+1; t<=t_wto; t++)
	{	  
	  if(fabs(caldata[t])>retval)
	    retval = fabs(caldata[t]);
	}
      
      printf("Year : ");
      for(int t=t_reform+1; t<=t_wto; t++)
	{
	  printf(" %d ",t+1971);
	}
      printf("\nError:");
      for(int t=t_reform+1; t<=t_wto; t++)
	{
	  printf(" %+0.2f",caldata[t]);
	}
      printf("\n");
    }
  else
    {
      printf("Errors:\n");
      retval = -HUGE_VAL;
      double avg_pre80 = 0.0;
      for(int t=3; t<=t_reform; t++)
	{
	  avg_pre80 += fabs(caldata[t])*fabs(caldata[t]);
	}
      avg_pre80 = sqrt(avg_pre80/((double)(t_reform-3+1)));
      retval= avg_pre80;
      printf("Avg 1974-1979 = %0.2f\n\n",avg_pre80);
      
      for(int t=t_reform+1; t<=t_data_max; t++)
	{
	  printf("%d ",t+1971);
	}
      printf("\n");
      for(int t=t_reform+1; t<=t_data_max; t++)
	{
	  printf("%+0.2f ",caldata[t]);
	  
	  if(fabs(caldata[t])>retval)
	    {
	      retval = fabs(caldata[t]);
	    }
	}
    }

  time(&stop);
  printf("\nIteration %d complete in %0.0f seconds. Max error/RMSE = %0.6f\n",iter,difftime(stop,start),retval);

  return retval;
}

void update_probs(int reform_flag)
{
  // update P(NNTR-->MFN) based on pre-1980 data
  double avg_pre80 = 0.0;
  for(int t=3; t<t_reform; t++)
    {
      avg_pre80 += caldata[t];
    }
  avg_pre80 = avg_pre80/((double)(t_reform-3));
  
  if(avg_pre80>0.0)
    {
      tpu_trans_mat[0][1][0] = tpu_trans_mat[0][1][0] * (1.0+log(1.0-fabs(avg_pre80))*tpu_prob_update_speed);
    }
  else
    {
      tpu_trans_mat[0][1][0] = tpu_trans_mat[0][1][0] * (1.0+log(1.0+fabs(avg_pre80))*tpu_prob_update_speed);
    }
  tpu_trans_mat[0][0][0] = 1.0-tpu_trans_mat[0][1][0];
      
  for(int t=1; t<NT; t++)
    {
      tpu_trans_mat[0][1][t] = tpu_trans_mat[0][1][0];
      tpu_trans_mat[0][0][t] = tpu_trans_mat[0][0][0];
    }

  // update P_t(MFN-->NNTR) before 2000 based on annual data from 1980-2001
  for(int t=t_reform+1; t<=t_data_max; t++)
    {
      double err = caldata[t];
      double tmp=0.0;
      if(err>0.0)
	{
	  tmp = tpu_trans_mat[1][0][t-1] * (1.0 + log(1.0+fabs(err)) * tpu_prob_update_speed);
	}
      else
	{
	  tmp = tpu_trans_mat[1][0][t-1] * (1.0 - log(1.0+fabs(err)) * tpu_prob_update_speed);
	}

      if(tmp<0.95)
	{
	  tpu_trans_mat[1][0][t-1]=tmp;
	}
      
      tpu_trans_mat[1][1][t-1] = 1.0-tpu_trans_mat[1][0][t-1];
      
      if(gsl_isnan(tpu_trans_mat[1][0][t-1]))
	{
	  printf("NaN prob detected!\n");
	}
    }
  
  for(int t=0; t<t_reform; t++)
    {
      tpu_trans_mat[1][0][t] = tpu_trans_mat[1][0][t_reform];
      tpu_trans_mat[1][1][t] = 1.0-tpu_trans_mat[1][0][t];
    }
  
  for(int t=t_data_max; t<NT; t++)
    {
      tpu_trans_mat[1][0][t] = tpu_trans_mat[1][0][t_data_max - 1];
      tpu_trans_mat[1][1][t] = tpu_trans_mat[1][1][t_data_max - 1];
    }
}

int calibrate_probs(int reform_flag)
{
  printf("Calibrating TPU probabilities to match estimated NNTR gap coefficients...\n");
  
  time_t start, stop;
  time(&start);

  //int iter=0;
  double maxdiff = +HUGE_VAL;
  do
    {
      iter++;
      maxdiff = calc_coeffs(reform_flag);

      if(maxdiff<0)
	{
	  printf("Error while computing coefficients! Exiting...\n");
	  return 1;
	}
      
      if(maxdiff<coeff_err_tol)
	break;

      update_probs(reform_flag);
    }
  while(iter < max_cal_iter);

  time(&stop);
  printf("\nCalibration complete in %0.0f seconds\n",difftime(stop,start));

  return 0;
}


/////////////////////////////////////////////////////////////////////////////
// 6. Main function and setup/cleanup
/////////////////////////////////////////////////////////////////////////////

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

  discretize_z();
  calc_death_probs();
  //random_draws();

  time(&stop);
  printf("Setup complete! Runtime = %0.0f seconds.\n",
	 difftime(stop,start));
	  
  return 0;
}

int det_analysis()
{
  printf("Solving and simulating deterministic model...\n");

  time_t start, stop;
  time(&start);

  if(solve_policies2_det())
    return 1;

  double expart_rate=0.0;
  double exit_rate=0.0;
  double new_size;
  double avg_5yr_gr;
  
  //simul2(0,&expart_rate,&exit_rate,&new_size,&avg_5yr_gr,1);
  //create_panel_dataset(0);

  //simul2(1,&expart_rate,&exit_rate,&new_size,&avg_5yr_gr,0);
  //create_panel_dataset(1);

  //simul2(2,&expart_rate,&exit_rate,&new_size,&avg_5yr_gr,0);
  //create_panel_dataset(2);

  do_trans_dyn(0);
  calc_cal_moments(&expart_rate,&exit_rate,&new_size);
  
  do_trans_dyn(1);
  do_trans_dyn(2);

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
  if(calibrate_probs(3))
    return 1;
  //calc_coeffs(3);
  //update_probs(3);
  
  // finish program
  linebreak();  
  time(&stop);
  printf("\nProgram complete! Total runtime: %0.16f seconds.\n",difftime(stop,start));

  return 0;
}


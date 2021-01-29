# tpu_akkrs

<b>Jan 28 update: Code simulates two forms of TPU. In the first, firms believe it will last forever, and then it ends unexpectedly after 20 periods. In the second, they know in advance that it will only last 20 periods. Added python script to process simulated data.</b>

The repo is organized into two main folders:
- "model" contains C code that simulates a panel of exporters
- "scripts" contains Python codes that process the simulated data and perform estimations

<h2>1. Model</h2>
The C code is organized into three folders: "src," which contains the source code; "bin," which contains the executable; and "output," which contains the programs output. The main "model" folder also contains a makefile used to create the executable.

Key ingredients
- 30 industries with pre-reform tariffs ranging from 0% to 100%. I would like to eventually use the actual data here, but we need to decide what constitutes an industry. Is it an HS6 code or a broader sector?
- 1000 firms (i.e. potential exporters) per industry
- 100 simulations, each with 200 periods. The first 100 periods of each simulation allow the model to converge to a pre-reform steady state. The second 100 periods capture the transition dynamics following a reform.
- Statistics like the export participation rate, exit rate, and new entrant size are computed for each simulation's steady state, then averaged across simulations. We can add more statistics and eventually calibrate to something as desired.
- I consider 3 reforms. The first assumes all tariffs go to zero with no probability of reversal. The second assumes all tariffs to to zero, but there is A 50% chance of reversal in each period. Firms believe this uncertainty will last forever, but it ends unexpectedly after 20 periods. The last reform is like the second, except that firms know in advance the uncertainty will only last 20 periods.
- For each reform, I create a dataset with the following columns: simulation number, industry, year, firm ID, tariff, TPU exposure (i.e., change in tariff), exports, entrant, exit, incumbent.

We will probably want to eventually do something different with a reversal probability that diminishes over time. I also am not sure how to capture the idea that there was essentially zero trade between the US and China before ~1980. According to Handley and Limao, the average decrease in tariffs was 30p.p. There might still be "too much" trade in any standard model with tariffs of 30-40%.

<h2>2. Scripts</h2>
The Python script proc_simul.py does the following:
- loads the simulated data for each of the three reforms
- aggregates exports and the number of exporters by year for each simulation
- computes means of these variables for each year across simulations to smooth out randomness
- plots the time series of these variables for each reform in simul_trans.pdf

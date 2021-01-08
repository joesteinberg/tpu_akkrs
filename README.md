# tpu_akkrs

The repo is organized into two main folders:
- "model" contains C code that simulates a panel of exporters
- "scripts" contains Python codes that process the simulated data and perform estimations

<h2>1. Model</h2>
The C code is organized into three folders: "src," which contains the source code; "bin," which contains the executable; and "output," which contains the programs output. The main "model" folder also contains a makefile used to create the executable.

Jan 8 update: The rough draft of the source code is finished. Here are the main features:
- 30 industries with pre-reform tariffs ranging from 0% to 100%. I would like to eventually use the actual data here, but we need to decide what constitutes an industry. Is it an HS6 code or a broader sector?
- 1000 firms (i.e. potential exporters) per industry
- 100 simulations, each with 200 periods. The first 100 periods of each simulation allow the model to converge to a pre-reform steady state. The second 100 periods capture the transition dynamics following a reform.
- Statistics like the export participation rate, exit rate, and new entrant size are computed for each simulation's steady state, then averaged across simulations. We can add more statistics and eventually calibrate to something as desired.
- I consider 2 reforms. The first assumes all tariffs go to zero with no probability of reversal. The second assumes all tariffs to to zero, but there is A 50% chance of reversal in each period. We will probably want to eventually do something different with a reversal probability that diminishes over time. I also am not sure how to capture the idea that there was essentially zero trade between the US and China before ~1980. According to Handley and Limao, the average decrease in tariffs was 30p.p. There might still be "too much" trade in any standard model with tariffs of 30-40%.
- For each reform, I create a dataset with the following columns: simulation number, industry, year, firm ID, tariff, exports, entrant, exit, incumbent (note I have commented out the calls to the function that creates the datasets as they are pretty large).

<h2>2. Scripts</h2>
TBA

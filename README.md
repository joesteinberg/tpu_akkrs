# tpu_akkrs

The repo is organized into two main folders:
- "model" contains C code that simulates a panel of exporters
- "scripts" contains Python codes that process the simulated data and perform estimations

<h2>1. Model</h2>
The C code is organized into three folders: "src," which contains the source code; "bin," which contains the executable; and "output," which contains the programs output. The main "model" folder also contains a makefile used to create the executable.

For now, there is a piece of placeholder code in the src folder that I created in another project. It implements a version of the new-exporter dynamics model of Alessandria, Choi, and Ruhl (2020). It needs to be modified in a few ways
- add industries
- remove firm fixed effects
- ....

<h2>2. Scripts</h2>
TBA

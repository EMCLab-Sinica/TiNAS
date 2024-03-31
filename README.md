# TiNAS : IMO Sensitive Neural Architecture Search

<!-- ABOUT THE PROJECT -->
## Overview

This repository contains TiNAS, an intermittent-aware neural architecture search (NAS) framework which balances intermittency management overhead (IMO) and accuracy. 
The DNN solutions can be feasibly deployed and executed on *intermittently-powered* systems. 
TiNAS leverages two key guidelines related to IMO sensitivity, i.e., the varying sensitivity towards IMO across different DNN characteristics. 

* Guideline 1: Reduce the search space by excluding architectural parameters with low IMO sensitivity, thus improving accuracy without increasing the IMO
* Guideline 2: Focus the search on network blocks with high IMO sensitivity, to quickly find an intermittent-friendly DNN with high accuracy and low IMO

We build TiNAS on top of the integration between two state-of-the-art NAS frameworks, namely [TinyNAS] (https://github.com/mit-han-lab/mcunet) for MCUs and [iNAS](https://github.com/EMCLab-Sinica/Intermittent-aware-NAS) for intermittent systems. 
We adapt the search space optimizer and evolutionary search strategy of TinyNAS to incorporate the above two guidelines. 


The derived TiNAS solutions are deployed on the Texas Instruments MSP430FR5994 LaunchPad and executed using an intermittent inference library that is also included in this repository. 

A demo video comparing solutions found by TiNAS and two variants (TiNAS-M: performing direct IMO minimization and TiNAS-U: unaware of IMO) can be found [here](https://youtu.be/ylc0P3ObJEg)



<!-- TABLE OF CONTENTS -->
## Table of Contents
* [Directory/File Structure](#directory/file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Setup and Build](#setup-and-build)
* [Using TiNAS](#using-tinas)
  


## Directory/File Structure
Below is an explanation of the key directories/files found in this repository. 

`TiNAS/NASBase/ss_optimization` contains the implementation for the search space optimizer (adapted from TinyNAS)<br/>
`TiNAS/NASBase/evo_search` contains the implementation for the evolutionary search strategy (adapted from TinyNAS)<br/>
`TiNAS/NASBase/hw_cost` contains the implementation for the intermittent inference cost model and intermittent execution design explorer (adapted from iNAS)<br/>
`TiNAS/NASBase/model` contains the search space definition, supernet and subnet structure<br/>
`TiNAS/tools/imo_sensitivity` contains the implementation for the IMO sensitivity analysis tool<br/>
`TiNAS/DNNDumper` is a helper module used to convert the derived solutions into a custom C data structure recognizable by the intermittent inference runtime library<br/>
`TiNAS/settings` contains the settings used for evaluation (for different datasets and baseline approaches)<br/>
`TiNAS/settings.py` contains the overall NAS settings and implementation for managing/loading settings files<br/>
`TiNAS/misc_scripts` contains miscellaneous helper scripts<br/>
`TiNAS/requirements.txt` contains the dependencies required to run TiNAS<br/>
`intermittent-inference-library` contains the intermittent inference runtime library developed for the TI-MSP430FR5994 (extended from iNAS's inference library)<br/>


## Getting Started

### Prerequisites

###### TiNAS
TiNAS is implemented using Python 3.7+, so we recommend installing the [Anaconda](https://docs.anaconda.com/anaconda/install/) distribution of Python and related packages. The required dependencies can be found in the included `TiNAS/requirements.txt` file. 

###### Intermittent Inference Library
Here is the basic software and hardware needed to build/run the intermittent inference runtime library. 
* [Code composer studio](http://www.ti.com/tool/CCSTUDIO "link") (recommended versions: > 12.0)
* [MSP Driver Library](http://www.ti.com/tool/MSPDRIVERLIB "link")
* [MSP DSP Library](http://www.ti.com/tool/MSP-DSPLIB "link")
* [MSP-EXP430FR5994 LaunchPad](http://www.ti.com/tool/MSP-EXP430FR5994 "link")

### Setup and Build

###### TiNAS
1. Download/clone this repository
2. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) Python distribution 
3. [Create and activate] (https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) virtual environment
4. Install dependencies `pip install -r requirements.txt`
5. Run TiNAS: <br/>
`python -m NASBase.run_nas --suffix cifar_example_run --settings settings/CIFAR10.json,settings/TiNAS-test.json --stages 1 --imcreq 0 --latreq 100 --ccap 0.005 --no-rlogger`

###### Intermittent Inference Library
1. Download/clone this repository
2. Download `Driver` & `DSP` library from http://www.ti.com/ 
3. Import this project to your workspace of code composer studio (CCS). 
4. Add `PATH_TO_DSPLIB` & `PATH_TO_DIRVERLIB` to library search path



# Report Due Date: May 06, 2024, 11:59PM EST

# Final Project Baselines and Example Designs
For a guide to the files, please see the workspace README.

## Final Project
The final project starting code is based off of the publicly-available
`timeloop-accelergy-exercises` repository, which contains example designs,
workloads, and tutorials for using Timeloop and Accelergy.

### Download Instructions

Please pull the docker first to update the container, and then start with `docker-compose up`. 
```
cd <your-git-repo-for-final-project>
export DOCKER_ARCH=<your arch amd64 or arm64>
docker-compose pull
docker-compose up
```
After finishing the project, please commit all changes and push back to this repository.

We recommend using VSCode to perform any advanced development work. The instructions
to set up VSCode with Docker can be found [here](https://https://code.visualstudio.com/docs/devcontainers/containers).

### Usage Instructions

BEFORE BEGINNING YOUR PROJECT, PLEASE READ THE FOLLOWING INSTRUCTIONS CAREFULLY. Failure to follow instructions may result in a zero for the assignment.

**Please add a license by filling in the LICENSE file. We recommend the [MIT open-source license](https://opensource.org/license/MIT).**

Before using any example designs, please go through the first two tutorials in the starter code.
These tutorials will help you understand how to use the tools. TAs will not provide help or answer questions if you have not
gone through these tutorials.

After going through the tutorials, you may use the provided example designs, layer
shapes, and scripts as a starting point for your final project. Please follow the
README files provided in the exercises repository. Note that subdirectories often
have nested README files.

For explanations of the inputs and outputs of the tools, see the
[Timeloop/Accelergy documentation](https://timeloop.csail.mit.edu/v4) and
[Timeloop/Accelergy tutorial](http://accelergy.mit.edu/tutorial.html). **Reading
documentation is an important skill and the TAs will not answer questions for
which answers may be found in the documentation.**

As described in the final project information slides, you can use the different
baselines for different final project ideas.
 
If your final project needs simulation/design support that is not currently
available in the provided framework/examples, please reach out to the course
staff **BEFORE** you get started on designing the project.

###  Submission instructions
Please copy all necessary files into this repository and commit them. If you'd like
to use files from the exercises, you may copy them or initialize the exercises as a
submodule.

###  Related reading

 - [Timeloop/Accelergy documentation](https://timeloop.csail.mit.edu/v4)
 - [Timeloop/Accelergy tutorial](http://accelergy.mit.edu/tutorial.html)
 - [SparseLoop tutorial](https://accelergy.mit.edu/sparse_tutorial.html)
 - [eyeriss-like design](https://people.csail.mit.edu/emer/papers/2017.01.jssc.eyeriss_design.pdf)
 - [simba-like architecture](https://people.eecs.berkeley.edu/~ysshao/assets/papers/shao2019-micro.pdf)
 - simple weight stationary architecture: you can refer to the related lecture notes
 - simple output stationary architecture: you can refer to the related lecture notes

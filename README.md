#  ON THE EFFECTIVENESS OF TASK GRANULARITY FOR TRANSFER LEARNING

A repo containing experiments to show the effect of task granularity on the performance of video classification and captioning. 
More details can be found in our paper:
https://arxiv.org/pdf/1804.09235.pdf

To understand the code structure of this repo, the inheritance and data flow diagrams are provided below.

##### Inheritance Diagram

![Inheritance Diagram](https://github.com/fmahdisoltani/misc2/blob/master/images/Inheritance_Diagram.jpg?raw=true "Inheritance Diagram")

##### Data Flow Diagram 
![Data Flow Diagram](https://github.com/fmahdisoltani/misc2/blob/master/images/Data_Flow_Diagram.jpg?raw=true "Data Flow Diagram")

#### How to Solve Unresolved Reference Issue in Pycharm with pybuilder

1. Mark the directory 'src/python' as source by right clicking that directory in the project view, going to 'Mark Directory as' and choosing 'Sources Root'

2. Add the directory to PYTHONPATH by going to File -> Settings -> Build, Execution, Deployment -> Console -> Python Console and checking the "Add source roots to PYTHONPATH" box

3. Run 'pyb pycharm\_generate'   

## Coding Standards

1. At least 50% line coverage
2. Commit python files as well as their tests
3. The person merging the pull request should not have taken part in developing it

### Pull Request (PR) Rules

1. Check the code does not break before a PR is created.
2. The power subset of the contributors should take care of the PR.
3. No requests about the original code, create an issue or PR instead.
4. Most (all if possible) changes should be requested in the first iteration.
5. Be vocal when you disagree.

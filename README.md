# ImageShap
CSCI 5922 Final Project: Combining LIME and SHAP to Generate More Interpretable SHAP Images

How important is a feature?
The answer is easy when asked of a linear model. In a linear model, each feature contributes to the output of the model in proportion to its value and the value of its assigned weight as in the linear function below.

<img src="https://github.com/lhayne/ImageShap/blob/master/docs/linearEquation.png" width="300" />

In a non-linear model, the question of feature importance becomes a little more difficult to answer. Shapley values provide an intuitive solution from game theory to measuring feature importance. Shapley values calculate the change in output of a "game" caused when a player is added to a game that already contains all subsets of other players. In this way, Shapley values effectively calculate feature importance, taking into account possible interaction effects between players. Below, the contribution of player A is calculated by subtracting how much the table earns without player A from the amount the table earns with them in all possible configurations.

<img src="https://github.com/lhayne/ImageShap/blob/master/docs/shap_payout.png" width="500" />

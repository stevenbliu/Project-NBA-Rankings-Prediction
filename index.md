#  NBA Rankings with Graph Neural Networks
## by Steven Liu and Aurelio Barrios
(Keep Simple-Scroll-Down website. Pictures, Plots. 'Informal Narrative')

![asdsa](https://cdn3.onlinegrad.syracuse.edu/content/fe1dbccb03e94a6c823c9737e8b5b368/4170_esportsupdate_viewers.jpg)
- The NBA is one of the most popular sports in the United States with 63 million viewers.
- The structure of the NBA resembles a graph and thus can be best represented with Graph Neural Networks.
- Using Graph Neural Networks, we will be able to make predictions of the ranking of each team.
- We believe that the accuracy of the Graph Neural Network model would be better than the current models.

# Why predict NBA Rankings?
![top10](img/top10.png)
- Here, we can see what teams need to achieve to become one of the top 10 teams in the NBA.
- Within sports and the NBA, statistics are crucial because they can tell you how well you are doing and what you are best or worst at. 
- Our model will be able to determine the impact of the changes made in a team's ranking. 
- This can help give direction to where you or your team needs to improve.

# Why Graph Nueral Networks?
- Most sports, such as the NBA, we can expect competitors to perform differently depending on who they're up against.
- The NBA regular season consists of the 81 games, and teams do not play each other the same number of times.
- This means that the ranking of the team can be influenced by the schedule that their given that season.
- Incorporating the regular season schedule within our model will allow us to capture the performance of each team with their unique match up.
- You can express the number of times each team plays each other with a weighted graph.
- This is why Graph Neural Networks work, and they are also one of the few models that can take advantage of this structure.

# How did we do it?
- 123


# Abstract
  The NBA contains many challenges when attempting to make predictions. The
performance of a team in the NBA is difficult because many things can happen over the
course of 81 games. Our analysis attempts to produce accurate results by exploiting the
natural structure of the NBA league and data of previous player stats. Our analysis
begins with identifying the features that show the highest correlation to the ranking of a
team, then we will take advantage of the schedule of each team to learn the unique
performance of a team against every other team. After taking advantage of the features
and the schedule of the teams, we expect to be able to make accurate predictions of
NBA seedings before a season starts.

  
# Introduction (What is the problem? Why do we care about it? What is our approach? Why is this our approach?)
  The NBA is one of the most popular sports in the U.S. It is the most followed
sports league on social media with more than 150 million followers. With this many
people keeping track of sport, we should expect that statistics on the sport will be useful
for many. An example is that we can apply our results from this project into making
sports bets. If we develop a model that can produce results with good accuracy, then
we can potentially use this model to place winning bets on the sport. Our assumption is
that the schedule of a team matters on their rankings in the season, this could make the
NBA management aware that the schedule is making the determination of the seedings
or likelihood of winning the competition unfair for some teams. This hasn’t been
something that has been addressed in U.S. sports yet despite similar concerns from
fans and casters. It is common knowledge that the ‘Eastern Conference’ has always
contained weaker teams compared to the ‘Western Conference’ which made
achievements from the ‘Eastern Conference’ less acknowledged. Hopefully, we can
make it more obvious that some of the ways the league conducts itself is unfair.
Most prior work in predicting NBA seedings are inaccurate and rely on examining
the data from the current season. NBA seedings predictions are heavily influenced by
the performance in the current season, but this also makes predictions less impressive.
In our work, we are able to take advantage of the large amount of data from previous
NBA seasons starting from 1946. This allows our model to have a plethora of data to
learn from which will allow us to make accurate predictions without looking at the
statistics from the season we want to predict.
The data we decided to use for our model will simply be player stats, team
rosters, and team schedules. With team rosters, we will be able to determine the
aggregate team stats with the player stats, and knowing the match ups between the
teams, we will be able to get an understanding of how difficult the season will be for
each team. The relationship between teams will be represented with an edge, and each
team will be a node. Each node will contain an aggregation of player stats, and we will
have a fully developed graph network to input into our graph neural network model.
We will be implementing two models to predict the seedlings of the teams. We
will apply the classical Graph Convolutional Network, GCN and GraphSAGE to our
network.

  
# Methods (Description of our methods on how it is effective, In-Depth explanation depends on target audience, Higher-Level than Report)
- We decided to use GraphSAGE because it works for its scalability on big data. With GraphSAGE we will be able to use this model to make predictions in other sports that have many more competitors. For example, golf, an individual sport which will have a significant larger number of nodes than team sports.


  When implementing the traditional GCN, the GCN model is initialized with the
number of nodes, n, number of hidden layers, l, and the number of classes, c. Inside the
GCN model, we have the GCN layers which are initialized with an adjacency matrix of
structure of the graph, A and the feature matrix, X. The shape of the adjacency matrix
should be n x n. Thus the feature matrix would be n x f, where f is the number of
features. We will use ReLU as the activation function, σ, of these layers and use a
softmax to make the classification. To train and test the model, we will need to call the
forward method of the GCN model, which will take an adjacency matrix of node edges,
and a feature matrix. There is a parameter in the GCN forward method to specify
whether you want to use Kipf & Welling’s normalization of the adjacency matrix, A, or to
leave it unnormalized.
The left formula shows how the layer will be computed without normalizing A, while the
right shows how it will be computed with normalizing A. The model can use cross
entropy loss to tune the weight parameters in each GCN layer. In our case, we will use
a categorical cross entropy loss.
GraphSAGE will create batches of neighborhoods and aggregate the information from
these neighborhoods into a new feature matrix. The forward method will take in a batch
of nodes, b and the depth size, k to develop a neighborhood. The model will then use b
and k to create a subsample of A, an adjacency matrix that defines the neighborhood of
every node in batch, b.
The batch of nodes, b and subsample of A will be then used as input to the GraphSAGE
aggregators, mean and pooling that can also be specified as a parameter in the forward
method.
The mean aggregator layer will output a feature matrix that contains the average
of neighborhood features. The pooling aggregator layer will output the original feature
matrix b, concatenated with the pooling of the neighborhood for each node in b. The
forward algorithm here should return a feature matrix for each node. In order to make
the algorithm more efficient, we can use matrix multiplication instead of a for loop in line
3.We will multiply the matrix of the last iteration, h, with A. This results in a matrix that
contains the sum of the features of every node. We can tweak this updated matrix
depending on which aggregator we decide to use. What makes the GraphSAGE loss
function interesting is that it only requires the output of the GraphSAGE model as input.
The loss function is calculated based on the similarity between the input, thus it doesn’t
need to know the true label of a node. It is also possible to use stochastic gradient
descent as the loss function. The parameters that will be tuned are the weights of the
aggregator layers. In order to implement LPA-GCN, we need to understand the
implementation of LPA and GCN. We will make the assumption that two connected
nodes are likely to have the same label, thus it propagates labels iteratively along the
edges. LPA will require the label distribution for each node, and A.
These are the two steps of the propagation rule. First we will use the normalized A to
propagate labels to their numbers. Then we will reset labeled nodes to the initial values.
When unifying LPA and GCN, we are unable to use Kipf & Welling’s normalization,
instead we use the one shown in the formula above. Thus when we apply the GCN
layer to our LPA-GCN model, it will look like:
With LPA, we can learn the optimal edge weight, A* by minimizing the predicted labels
of LPA:
The same can be done to find the optimal weight matrix, W*.
With A*, we will also be able to find the optimal D, D*. Then our optimized and
normalized GCN layer will look like:
Where W will be replaced with the optimal W, W*. Theoretically, this is an improved
GCN model and we would expect better performance results.
  
# Results (End with Resuls and Impact)
  stuff stuff stuff
  
# Conclusion (What we learned, Biggest Part)
  stuff stuff stuff
  
# Appendix
Project Proposal Statement
Sports data can have a structure that can be taken advantage of with graph
networks. In sports, how well a team performs will vary on the team they are playing
against. This relationship will be the schedule of a team, which determines which teams
play which along with how many times they will play each other in a given season. With
the schedule, we will be able to develop a model that captures the performance of each
team in their respective match up. With this model, we believe that we will be able to
deliver higher accuracy when predicting the rankings of a team. It is helpful to be able to
predict the rankings of a team because it will allow us to determine what teams will
make it into the playoffs based on previous data. The data we can use to make these
predictions is from the stats of each player on the team. We will also develop an option
that allows to simulate trading of players. This way our project will be able to determine
how much better or worse a team will become after a given trade. It could possibly help
teams determine whether a trade is good or bad. This would most likely be the primary
feature we will develop and expected to be used on the website.












You can use the [editor on GitHub](https://github.com/sdevinl/DSC180B_Project/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

Example:
![Jetpacktocat](https://octodex.github.com/images/jetpacktocat.png)
![top10](img/top10.png)




For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/sdevinl/DSC180B_Project/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.

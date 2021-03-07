#  NBA Rankings with Graph Neural Networks
## by Steven Liu and Aurelio Barrios
(Keep Simple-Scroll-Down website. Pictures, Plots. 'Informal Narrative')

![viewers](https://cdn3.onlinegrad.syracuse.edu/content/fe1dbccb03e94a6c823c9737e8b5b368/4170_esportsupdate_viewers.jpg)
- The NBA is one of the most popular sports in the United States with **63 million viewers**.
- The structure of the NBA resembles a graph and thus can be best represented with Graph Neural Networks.
- Using Graph Neural Networks, we will be able to make predictions of the ranking of each team.
- We believe that the accuracy of the Graph Neural Network model would be better than the [_current models_](https://fivethirtyeight.com/features/introducing-raptor-our-new-metric-for-the-modern-nba/).

# What's wrong with Prior Work?
- [**FiveThirtyEight**](https://fivethirtyeight.com/tag/nba/) is a popular analytics website that has their own NBA ranking prediction model. 
- It specifically mentions that the model doesn't account account for wins and losses of each team.
- The projection is entirely based on an estimate of each player's future performance based on other similar NBA players.
- This model doesn't look at the relationship between teams, which is the primary advantage our model will have.

# Why predict NBA Rankings?
![top10](img/top10.png)
Source: [https://watchstadium.com/which-nba-statistics-actually-translate-to-wins-07-13-2019/](https://watchstadium.com/which-nba-statistics-actually-translate-to-wins-07-13-2019/)
- Here, we can see what teams need to achieve to become one of the top 10 teams in the NBA.
- Within sports and the NBA, **statistics are crucial because they can tell you how well you are doing and what you are best or worst at**. 
- Our model will be able to determine the impact of the changes made in a team's ranking. 
- This can help give direction to where you or your team needs to improve.

# Why Graph Neural Networks (GNNs)?
![graph](img/graph.png)
- Most sports, such as the NBA, we can expect competitors to perform differently depending on who they're up against.
- The NBA regular season consists of the 81 games, and teams do not play each other the same number of times.
- This means that the **ranking of the team can be influenced by the schedule that their given that season**.
- Incorporating the regular season schedule within our model will allow us to capture the performance of each team with their unique match up.
- You can express the number of times each team plays each other with a weighted graph.
- This is why Graph Neural Networks work, and they are also one of the few models that can take advantage of this structure.

# What's our data?
![data](https://user-images.githubusercontent.com/45984322/110225932-ecefc080-7e9e-11eb-937d-bed63d1d6786.png)
- The data we will be using will be **individual player statistics, team rosters, and team schedules from the last 10 seasons**.
- All the data will be webscraped from https://www.basketball-reference.com/.
- After webscraping the data, we cannot directly input it into our model.
- We will need to develop a data pipeline to preprocess it for our Graph Neural Network!

# Which Graph Neural Network?
![image](https://user-images.githubusercontent.com/45984322/110225963-317b5c00-7e9f-11eb-82a4-4eae23767c17.png)
- There are many GNNs in the field, but the one we will use GraphSAGE.
- GraphSAGE is a framework for inductive representation learning on large graphs.
- The steps of GraphSAGE are:
![image](https://user-images.githubusercontent.com/45984322/110226000-b49cb200-7e9f-11eb-9dd2-579df19259fe.png)
- There are multiple techniques to aggregate the feature information from neighbors, such as mean, mean-pooling, max-pooling, and sequential.
- GraphSAGE is scalable to node attributes, thus we will be able to experiment with as many features as we'd like for our player stats.
- Each node in our graph network will be a team, which will consist of the aggregate of all player stats on the team's roster. 
- Thus an input of GraphSAGE will require the use of the **player statistics** and **team rosters**, the feature matrix.
- To determine the neighbors of the graph network we will use the **team schedules**.
- This will be our second input of GraphSAGE, the adjacency matrix.
- With this, we can take advantage of all GraphSAGE has to offer!

# What are we doing?
- **We will be inputting player stats, team rosters, and team schedules into our GraphSAGE model to output the ranking for each team.**


# Challenges & Solutions
1) Scraping our own data was difficult. Stats were hidden under an on-hover button, which we couldn't access.
   - We were able to find a third-party scraper to get the data we needed. Thanks to [https://pypi.org/project/basketball-reference-scraper/)](https://pypi.org/project/basketball-reference-scraper/).

2) We were obtaining extremely low accuracies with the GraphSAGE model.
   - We thought our implemenation of GraphSAGE was incorrect, so we decided to use the author's at [https://github.com/twjiang/graphSAGE-pytorch](https://github.com/twjiang/graphSAGE-pytorch). This did not change anything. 
   - This was mainly due to the model's loss function penalizing wrong predictions too much.
   - Making predictions that are close to the correct correct seeding should be okay. 
   - To combat this, instead of the labels being from 1-30 for each team, we decided to make them from 0-1, whether the team made it to the playoffs or not. 
   - This reduces the influence of the loss function and we were able to obtain good results!
 
3) However our goal isn't to predict whether teams make it to the playoffs, it is to predict their actual rankings!
   - To do this, we had to modify the output of the GraphSAGE model.
   - Instead of outputting labels, we had the model output the probabilities for each label.
   - We then used the probabilities as the ranking for each team.
   - It was a success, still able to obtain good results.
   
4) All the model told us was what accuracy, it is all useless unless we know exactly what teams placed where!
   - This required more modification to the model.
   - The aggregators and batching of the model resulted in some shuffling of the data which made it difficult to know which team placed what seed!
   - (Currently working on this)




# Our Results!
   ( Many plots with accuracies and comparisons to other model and aggregators coming!)
   
  [Loss][results/resultsModelsLoss.png]

# Conclusion!
   (Waiting for Results)

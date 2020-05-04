# MovieLens Capstone Project R Code
# Shilpa Susan Thomas

# Data Preparation

#Load the R packages that will be required to run the code for the project and will be useful for analysis and visualisations
library(tidyverse)
library(dslabs)
library(caret)
library(data.table)
library(lubridate)
library(ggplot2)


################################
# Create edx set, validation set
################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Data Exploration

#the training set
# intial 7 rows with header
head(edx)
# basic summary statistics
nrow(edx)
ncol(edx)
summary(edx)
edx %>% summarize(n_movies = n_distinct(movieId))
edx %>% summarize(n_users = n_distinct(userId))
drama <- edx %>% filter(str_detect(genres,"Drama"))
comedy <- edx %>% filter(str_detect(genres,"Comedy"))
thriller <- edx %>% filter(str_detect(genres,"Thriller"))
romance <- edx %>% filter(str_detect(genres,"Romance"))
nrow(drama)
nrow(comedy)
nrow(thriller)
nrow(romance)
edx %>% group_by(title) %>% summarise(number = n()) %>%
  arrange(desc(number))
#modify datasets to make the year of the movie as a separate column
edx <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
#check if year has been added correctly
head(edx)

#some data visualisations

#load some extra R packages that will be needed for analysis and visualisations
library(dplyr)
library(gridExtra)
library(knitr)
library(kableExtra)

#Plot of Ratings per movie
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie")

#Plot of Ratings per user
edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  xlab("Number of ratings") + 
  ylab("Number of users") +
  ggtitle("Number of ratings per user")

#Plot of Ratings Distribution
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "black") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("The distribution of user ratings")

#Plot of Ratings over time
edx %>% 
  group_by(year)%>%summarize(rating=mean(rating)) %>%
  ggplot(aes(year,rating))+geom_point()+geom_smooth() +
  xlab('Movie Release Year') + 
  ggtitle("Ratings according to release date of movie")

#Modelling Process

#RMSE function definition
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Partitioning of the edx dataset into separate training and test sets to design and test algorithm
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

#Average Model 

mu <- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating, mu)
rmse_results <- data.frame(method = "Just the Average Model", RMSE = naive_rmse)
rmse_results %>% knitr::kable() %>% kable_styling(position = 'center')%>%
  column_spec(1,border_left = T)%>%column_spec(2,border_right = T)%>%
  row_spec(0,bold=T)


#Movie Effect Model
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings_movie <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_1_rmse <- RMSE(predicted_ratings_movie, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable() %>% kable_styling(position = 'center')%>%
  column_spec(1,border_left = T)%>%column_spec(2,border_right = T)%>%
  row_spec(0,bold=T)

#Movie + User Effect Model
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings_user <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings_user, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))

rmse_results %>% knitr::kable() %>% kable_styling(position = 'center')%>%
  column_spec(1,border_left = T)%>%column_spec(2,border_right = T)%>%
  row_spec(0,bold=T)

#Regularised Movie + User Effect Model

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

lambda1 <- lambdas[which.min(rmses)]
lambda1

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularised Movie + User Effect Model",  
                                     RMSE = min(rmses)))

rmse_results %>% knitr::kable() %>% kable_styling(position = 'center')%>%
  column_spec(1,border_left = T)%>%column_spec(2,border_right = T)%>%
  row_spec(0,bold=T)

#Regularised Movie + User + Year Effect Model

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_y <- train_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_y, by = 'year') %>%
    mutate(pred = mu + b_i + b_u + b_y) %>% 
    .$pred
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

lambda2 <- lambdas[which.min(rmses)]
lambda2

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularised Movie + User + Year Effect Model",  
                                     RMSE = min(rmses)))

rmse_results %>% knitr::kable() %>% kable_styling(position = 'center')%>%
  column_spec(1,border_left = T)%>%column_spec(2,border_right = T)%>%
  row_spec(0,bold=T)

# Final Model using whole edx dataset and validation dataset to predict

mu <- mean(edx$rating)

b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda2), n_i = n()) 

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda2), n_u = n()) 

b_y <- edx %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+lambda2), n_y = n()) 

predicted_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_y, by = 'year') %>%
  mutate(pred = mu + b_i + b_u + b_y) %>% 
  .$pred

model_final_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Final Prediction Model using validation set",  
                                     RMSE = model_final_rmse ))
rmse_results %>% knitr::kable() %>% kable_styling(position = 'center')%>%
  column_spec(1,border_left = T)%>%column_spec(2,border_right = T)%>%
  row_spec(0,bold=T)








                  
                  
###################################################################################### 
#Title    : MovieLens Project                                                        #
#Author   : Delpagodage Ama Nayanahari Jayaweera                                     #
#Subtitle : Data Science: Capstone Project for Harvardx Professional Data Science    #
#           Certificate (MovieLens Project)                                          #
#Date     : 2024-12-05                                                               #
######################################################################################

#################################################
# MovieLens Rating Prediction Project Code      #
#################################################

# Note: this process could take a couple of minutes

######################################################################################
#Loading and Preparing the Data                                                      #
#The dataset is from the MovieLens 10M dataset (with 10 million ratings), which      #
#is downloaded and unzipped.                                                         #
#The ratings and movies data are loaded, cleaned, and merged into one dataframe      #
#movielens.                                                                          #
#The dataset is split into a training (train_set) and test (test_set) set, where 20% #
#of the data is used for testing.                                                    #
######################################################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

data_dir <- "D:/Edex/Final/Data_Science_Capstone_MovieLens/ml-10M100K"
ratings_file <- file.path(data_dir, "ratings.dat")
movies_file <- file.path(data_dir, "movies.dat")

dl <- "ml-10M100K.zip"
if (!file.exists(dl)) {
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  unzip(dl, exdir = data_dir)
}

if (!file.exists(ratings_file) || !file.exists(movies_file)) {
  stop("Required data files are missing. Please check file paths.")
}

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")


# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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


# Preview data
head(edx,3)
head(validation,3)

# Number of user-movie combinations = entries in the user movie matrix
n_distinct(edx$userId)*n_distinct(edx$movieId)

# Sparsity of the user movie matrix
round(nrow(edx)/(n_distinct(edx$userId)*n_distinct(edx$movieId))*100,1)



######################################################################################
# RMSE FUNCTION DEFINITION                                                           #
######################################################################################

# Define the RMSE function that is used to compute model performance
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}



######################################################################################
# DEFINE TRAIN AND TEST SET IN THE EDX DATASET                                       #
######################################################################################

# Test set will be 20% of edx dataset
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)


######################################################################################
# BASELINE MODEL                                                                     #
# A simple baseline model is built using the average rating (mu), movie effect (b_i),#
# and user effect (b_u).                                                             #
# The root mean square error (RMSE) is calculated on the test set as a performance   # 
# measure.                                                                           #
######################################################################################

# Average of ratings in train set
mu <- mean(train_set$rating)

# Adding the movie effect b_i
b_i <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu)) # calculate average rating for each movie

# Adding the user effect b_u
b_u <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i)) # calculate average rating for each user

# Making predictions on test set
predictions <- test_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

baseline_rmse <- RMSE(predictions, test_set$rating) # store baseline RMSE
print(paste("Baseline RMSE is",round(baseline_rmse,5))) # print baseline RMSE


######################################################################################
# EXPLORATORY ANALYSIS                                                               #
######################################################################################

######################################################################################
#The script generates some exploratory analysis to better understand the data:       #
# 1.Distribution of residuals(difference between true ratings and model predictions).#
# 2.Relationships between residuals and factors like the rating month, movie release #
#   year, and genres.                                                                #
######################################################################################

# Calculate and plot the residuals from baseline model
#-----------------------------------------------------

# Add a residuals column to train set
train_set <- train_set %>% 
  left_join(b_i, by = 'movieId') %>% 
  left_join(b_u, by = 'userId') %>% 
  mutate(residuals = rating - mu - b_i - b_u) %>% 
  select(-b_i, -b_u) 


# Plot a histogram of residuals
train_set %>% 
  ggplot(aes(residuals)) + geom_histogram() + theme_bw()


# Visual analysis of other variables
# ----------------------------------

# Explore rating date effect
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
library(lubridate)
train_set %>% 
  mutate(date=round_date(as_datetime(timestamp),unit="month")) %>% # transform timestamp and rounf to the month
  group_by(date) %>% 
  summarize(date=date[1], avg_res=mean(residuals), se_res= sd(residuals)/sqrt(n())) %>% # average rating for each month of rating
  ggplot(aes(date, avg_res, alpha=1/se_res)) + # create and store plot
  geom_point(show.legend = F) +
  ylim(c(-0.25, 0.75))+
  theme_bw() +
  theme(axis.title.y = element_blank()) -> plot_date

# Explore the movie release year effect
train_set %>% 
  mutate(year=as.numeric(str_extract(title,"(?<=\\()\\d{4}(?=\\))"))) %>% # extract year from the title column
  group_by(year) %>% 
  summarize(year=year[1], avg_res=mean(residuals), se_res = sd(residuals)/sqrt(n())) %>% # average rating for each year
  ggplot(aes(year, avg_res, alpha=1/se_res))+ # create and store plot
  geom_point(show.legend = F)+
  ylim(c(-0.25, 0.75))+
  theme_bw()+
  theme(axis.title.y = element_blank()) -> plot_year

# Explore genre effect
genres_list <- unique(train_set$genres) # extract all unique genres

train_set %>% 
  group_by(genres) %>% 
  summarize(genres=genres[1], n_ratings=n(), avg_res=mean(residuals), # average rating for each genre
            se_res=ifelse(n()>1,sd(residuals)/sqrt(n()),1000)) %>% 
  mutate(genresId=match(genres,genres_list)) %>% # transform genre name in an ID
  ggplot(aes(genresId, avg_res, alpha=1/se_res))+ # create and store plot
  geom_point(show.legend = F)+
  ylim(c(-0.25, 0.75))+
  theme_bw() +
  theme(axis.title.y = element_blank()) -> plot_genres

# Plots next to each other to compare
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
library(gridExtra)
grid.arrange(plot_date, plot_year, plot_genres, nrow=1, left="Residuals")


######################################################################################
# MY MODEL #1                                                                        #
# In this step, additional effects like the movie genre (b_k) and release year (b_n) #
# are added to the model.                                                            #
# The predictions are made on the test set, and the RMSE is calculated.              #
######################################################################################

# Clearing the predictions from old models
rm(predictions)

# Adding the genres effect b_k
b_k <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(genres) %>%
  summarize(b_k = mean(rating - mu - b_i - b_u)) # average rating for each genre

# Adding the release year effect b_n
b_n <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_k, by='genres') %>% 
  mutate(year=as.factor(str_extract(title,"(?<=\\()\\d{4}(?=\\))"))) %>% # extract year from title column
  group_by(year) %>%
  summarize(b_n = mean(rating - mu - b_i - b_u - b_k)) # average rating for each year

# Making predictions on test set
predictions <- test_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_k, by='genres') %>% 
  mutate(year=as.factor(str_extract(title,"(?<=\\()\\d{4}(?=\\))"))) %>% # extract year from title column
  left_join(b_n, by='year') %>% 
  mutate(pred = mu + b_i + b_u + b_k + b_n) %>%
  pull(pred)

mymodel_1_rmse <- RMSE(predictions, test_set$rating) # store model #1 RMSE
print(paste("RMSE with my model #1 is",round(mymodel_1_rmse,5))) # print model #1 RMSE


#########################
# EXPLORE REGULARIZATION
#########################

# Update the residuals column in train set
train_set <- train_set %>% 
  left_join(b_i, by = 'movieId') %>% 
  left_join(b_u, by = 'userId') %>% 
  left_join(b_k, by = 'genres') %>%
  mutate(year=as.factor(str_extract(title,"(?<=\\()\\d{4}(?=\\))"))) %>%
  left_join(b_n, by = 'year') %>% 
  mutate(residuals = rating - mu - b_i - b_u - b_k - b_n) %>% 
  select(-b_i, -b_u, -b_k, -b_n) 


# Visual analysis of the residuals
#----------------------------------

# Remove old plots
rm(plot_year, plot_genres, plot_date)

# Creating the plots: residuals against number of ratings
train_set %>% 
  group_by(movieId) %>% # movie effect
  summarize(n_ratings=n(), avg_res=mean(residuals)) %>% 
  ggplot(aes(n_ratings, avg_res)) + # create and store the plot
  geom_point() +
  ylim(c(-1,2))+
  theme_bw() +
  theme(axis.text.x = element_text(angle=90, hjust=1),
        axis.title.x = element_blank(),
        axis.title.y = element_blank()) +
  ggtitle("Movie effect")-> plot_movie 

train_set %>% 
  group_by(userId) %>% # user effect 
  summarize(n_ratings=n(), avg_res=mean(residuals)) %>% 
  ggplot(aes(n_ratings, avg_res)) + # create and store the plot
  geom_point() + 
  ylim(c(-1,2))+
  theme_bw() +
  theme(axis.text.x = element_text(angle=90, hjust=1),
        axis.title.x = element_blank(),
        axis.title.y = element_blank())+
  ggtitle("User effect")-> plot_user

train_set %>% 
  group_by(genres) %>% # genre effect
  summarize(n_ratings=n(), avg_res=mean(residuals)) %>% 
  ggplot(aes(n_ratings, avg_res)) + # create and store the plot
  geom_point() + 
  ylim(c(-1,2))+
  theme_bw() +
  theme(axis.text.x = element_text(angle=90, hjust=1),
        axis.title.x = element_blank(),
        axis.title.y = element_blank())+
  ggtitle("Genre effect")-> plot_genres

train_set %>% 
  group_by(year) %>% # movie release year effect
  summarize(n_ratings=n(), avg_res=mean(residuals)) %>% 
  ggplot(aes(n_ratings, avg_res)) + # create and store the plot
  geom_point() + 
  ylim(c(-1,2))+
  theme_bw() +
  theme(axis.text.x = element_text(angle=90, hjust=1),
        axis.title.x = element_blank(),
        axis.title.y = element_blank())+
  ggtitle("Year effect")-> plot_year

# plots next to each other
library(gridExtra)
grid.arrange(plot_movie, plot_user, plot_genres, plot_year, nrow=1, left="Residuals", bottom="Number of ratings")


# Example - movies with only 1 rating
# ------------------------------------

# Number of movies with only 1 rating
train_set %>% 
  group_by(movieId) %>% 
  summarize(n_ratings=n()) %>% 
  filter(n_ratings==1) %>% 
  nrow(.) -> one_rating

one_rating_percent <- one_rating/n_distinct(train_set$movieId)*100



######################################################################################
# MY MODEL #2                                                                        #
# The model is adjusted with regularization applied specifically to the movie        #
# effect (b_i), and the best lambda (regularization strength) is chosen by tuning the# 
# RMSE.                                                                              #
######################################################################################


# Regularization for movie effect
lambdas <- seq(0, 10, 0.25) # apply a set of lambdas

# function to tune the penalization term (lambda)
rmses <- sapply(lambdas, function(l){ 
  b_i_reg <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i_reg = sum(rating - mu)/(n()+l)) # penalization term applies here
  b_u_2 <- train_set %>%
    left_join(b_i_reg, by = 'movieId') %>% 
    group_by(userId) %>%
    summarize(b_u_2 = mean(rating - b_i_reg - mu))
  b_k_2 <- train_set %>%
    left_join(b_i_reg, by = 'movieId') %>% 
    left_join(b_u_2, by='userId') %>% 
    group_by(genres) %>%
    summarize(b_k_2 = mean(rating - b_i_reg - b_u_2 - mu))
  b_n_2 <- train_set %>%
    left_join(b_i_reg, by = 'movieId') %>% 
    left_join(b_u_2, by='userId') %>% 
    left_join(b_k_2, by='genres') %>% 
    group_by(year) %>%
    summarize(b_n_2 = mean(rating - b_i_reg - b_u_2 - b_k_2 - mu))
  
  predicted_ratings <- # predictions on the test set
    test_set %>% 
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_2, by='userId') %>% 
    left_join(b_k_2, by='genres') %>%
    mutate(year=as.factor(str_extract(title,"(?<=\\()\\d{4}(?=\\))"))) %>%
    left_join(b_n_2, by='year') %>% 
    mutate(pred = mu + b_i_reg + b_u_2 + b_k_2 + b_n_2) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

# plot the results of tuning lambda
tibble(lambdas=lambdas, rmses=rmses) %>% 
  ggplot(aes(lambdas, rmses)) +
  geom_point()+
  theme_bw()

# extract model results
l_i <- lambdas[which.min(rmses)] # lambda that minimizes RMSE
mymodel_2_rmse <- min(rmses) # minimum RMSE obtained during tuning
print(paste("Optimal lambda is", l_i)) # print optimal lambda
print(paste("RMSE with my model #2 is",round(mymodel_2_rmse,5))) # print model #2 RMSE



######################################################################################
# MY MODEL #3                                                                        #
#Regularization is extended to all effects (movie, user, genre, and release year),   #
# and the optimal lambda is determined.                                              #
######################################################################################


# Regularization for all effects
lambdas <- seq(0, 10, 0.25)# apply a set of lambdas

# function to tune the penalization term (lambda)
rmses <- sapply(lambdas, function(l){
  b_i_reg <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i_reg = sum(rating - mu)/(n()+l)) # penalization term applies here
  b_u_reg <- train_set %>% 
    left_join(b_i_reg, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_reg = sum(rating - b_i_reg - mu)/(n()+l)) # penalization term applies here
  b_k_reg <- train_set %>% 
    left_join(b_i_reg, by="movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_k_reg = sum(rating - b_i_reg - b_u_reg - mu)/(n()+l)) # penalization term applies here
  b_n_reg <- train_set %>% 
    left_join(b_i_reg, by="movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    left_join(b_k_reg, by="genres") %>% 
    mutate(year=as.factor(str_extract(title,"(?<=\\()\\d{4}(?=\\))"))) %>%
    group_by(year) %>%
    summarize(b_n_reg = sum(rating - b_i_reg - b_u_reg - b_k_reg - mu)/(n()+l)) # penalization term applies here
  
  predicted_ratings <- # make predictions on the test set
    test_set %>% 
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    left_join(b_k_reg, by='genres') %>% 
    mutate(year=as.factor(str_extract(title,"(?<=\\()\\d{4}(?=\\))"))) %>%
    left_join(b_n_reg, by='year') %>% 
    mutate(pred = mu + b_i_reg + b_u_reg + b_k_reg + b_n_reg) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})


# plot the results of tuning lambda
tibble(lambdas=lambdas, rmses=rmses) %>% 
  ggplot(aes(lambdas, rmses)) +
  geom_point()+
  theme_bw() 

# extract model results
lambda <- lambdas[which.min(rmses)] # lambda that minimizes RMSE
mymodel_3_rmse <- min(rmses) # minimum RMSE obtained during tuning
print(paste("Optimal lambda is", lambda)) # print optimal lambda
print(paste("RMSE with my model #3 is",round(mymodel_3_rmse,5))) # print model #3 RMSE


#########################################################################################
# FINAL VALIDATION                                                                      #
# The final models are applied to the validation set (validation), and the RMSE is      #
# calculated for each model.                                                            #
# Baseline Model RMSE is computed using mu_f, b_i_f, and b_u_f.                         #
# Model #1 RMSE is computed using b_k_f and b_n_f.                                      #
# The results will show how well each model generalizes to unseen data (validation set).#
#########################################################################################

# Apply models on the validation set

rm(predictions) # Clearing predictions from old model

# Baseline model - append _f suffix for 'final'
#-----------------------------------------------

mu_f <- mean(edx$rating) # average rating in edx set

# movie effect
b_i_f <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i_f = mean(rating - mu_f))

# user effect
b_u_f <- edx %>% 
  left_join(b_i_f, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u_f = mean(rating - mu_f - b_i_f))

# make baseline predictions on the validation set
predictions_baseline <- validation %>% 
  left_join(b_i_f, by='movieId') %>%
  left_join(b_u_f, by='userId') %>%
  mutate(pred = mu_f + b_i_f + b_u_f) %>%
  pull(pred)

baseline_rmse_f <- RMSE(predictions_baseline, validation$rating) # store baseline RMSE


# Model #1 - append _f suffix for 'final'
#----------------------------------------

# genre effect
b_k_f <- edx %>% 
  left_join(b_i_f, by='movieId') %>%
  left_join(b_u_f, by='userId') %>%
  group_by(genres) %>%
  summarize(b_k_f = mean(rating - mu_f - b_i_f - b_u_f))

# movie release year effect
b_n_f <- edx %>% 
  left_join(b_i_f, by='movieId') %>%
  left_join(b_u_f, by='userId') %>%
  left_join(b_k_f, by='genres') %>% 
  mutate(year=as.factor(str_extract(title,"(?<=\\()\\d{4}(?=\\))"))) %>% 
  group_by(year) %>%
  summarize(b_n_f = mean(rating - mu_f - b_i_f - b_u_f - b_k_f))

# make predictions on the validation set
predictions_model1 <- validation %>% 
  left_join(b_i_f, by='movieId') %>%
  left_join(b_u_f, by='userId') %>%
  left_join(b_k_f, by='genres') %>% 
  mutate(year=as.factor(str_extract(title,"(?<=\\()\\d{4}(?=\\))"))) %>% 
  left_join(b_n_f, by='year') %>% 
  mutate(pred = mu_f + b_i_f + b_u_f + b_k_f + b_n_f) %>%
  pull(pred)

mymodel_1_rmse_f <- RMSE(predictions_model1, validation$rating) # store model #1 RMSE


# Model #2 - append _f2 suffix for 'final #2'
#--------------------------------------------
###########################################################################################
#The model is adjusted with regularization applied specifically to the movie effect (b_i),#
#and the best lambda (regularization strength) is chosen by tuning the RMSE.              #
###########################################################################################

# Regularization of movie effect
b_i_f2 <- edx %>%
  group_by(movieId) %>%
  summarize(b_i_f2 = sum(rating - mu_f)/(n()+l_i)) # use tuned lambda here
b_u_f2 <- edx %>%
  left_join(b_i_f2, by = 'movieId') %>% 
  group_by(userId) %>%
  summarize(b_u_f2 = mean(rating - b_i_f2 - mu_f))
b_k_f2 <- edx %>%
  left_join(b_i_f2, by = 'movieId') %>% 
  left_join(b_u_f2, by='userId') %>% 
  group_by(genres) %>%
  summarize(b_k_f2 = mean(rating - b_i_f2 - b_u_f2 - mu_f))
b_n_f2 <- edx %>%
  left_join(b_i_f2, by = 'movieId') %>% 
  left_join(b_u_f2, by='userId') %>% 
  left_join(b_k_f2, by='genres') %>% 
  mutate(year=as.factor(str_extract(title,"(?<=\\()\\d{4}(?=\\))"))) %>%
  group_by(year) %>%
  summarize(b_n_f2 = mean(rating - b_i_f2 - b_u_f2 - b_k_f2 - mu_f))

# make predictions on the validation set
predictions_model2 <- validation %>% 
  left_join(b_i_f2, by='movieId') %>%
  left_join(b_u_f2, by='userId') %>%
  left_join(b_k_f2, by='genres') %>% 
  mutate(year=as.factor(str_extract(title,"(?<=\\()\\d{4}(?=\\))"))) %>% 
  left_join(b_n_f2, by='year') %>% 
  mutate(pred = mu_f + b_i_f2 + b_u_f2 + b_k_f2 + b_n_f2) %>%
  pull(pred)

mymodel_2_rmse_f <- RMSE(predictions_model2, validation$rating) # store model #2 RMSE


# Model #3 - append _f3 suffix for 'final #3'
#--------------------------------------------
###########################################################################################
#Regularization is extended to all effects (movie, user, genre, and release year),        #
#and the optimal lambda is determined.                                                    #
###########################################################################################

b_i_f3 <- edx %>%
  group_by(movieId) %>%
  summarize(b_i_f3 = sum(rating - mu_f)/(n()+lambda)) # use tuned lambda here
b_u_f3 <- edx %>%
  left_join(b_i_f3, by = 'movieId') %>% 
  group_by(userId) %>%
  summarize(b_u_f3 = sum(rating - b_i_f3 - mu_f)/(n()+lambda)) # use tuned lambda here
b_k_f3 <- edx %>%
  left_join(b_i_f3, by = 'movieId') %>% 
  left_join(b_u_f3, by='userId') %>% 
  group_by(genres) %>%
  summarize(b_k_f3 = sum(rating - b_i_f3 - b_u_f3 - mu_f)/(n()+lambda)) # use tuned lambda here
b_n_f3 <- edx %>%
  left_join(b_i_f3, by = 'movieId') %>% 
  left_join(b_u_f3, by='userId') %>% 
  left_join(b_k_f3, by='genres') %>%
  mutate(year=as.factor(str_extract(title,"(?<=\\()\\d{4}(?=\\))"))) %>%
  group_by(year) %>%
  summarize(b_n_f3 = sum(rating - b_i_f3 - b_u_f3 - b_k_f3 - mu_f)/(n()+lambda)) # use tuned lambda here

# make predictions on the validation set
predictions_model3 <- validation %>% 
  left_join(b_i_f3, by='movieId') %>%
  left_join(b_u_f3, by='userId') %>%
  left_join(b_k_f3, by='genres') %>% 
  mutate(year=as.factor(str_extract(title,"(?<=\\()\\d{4}(?=\\))"))) %>% 
  left_join(b_n_f3, by='year') %>% 
  mutate(pred = mu_f + b_i_f3 + b_u_f3 + b_k_f3 + b_n_f3) %>%
  pull(pred)

mymodel_3_rmse_f <- RMSE(predictions_model3, validation$rating) # stor model #3 RMSE


# create a tibble to store and compare results of the different models
rmse_results <- tibble(model = c("Baseline", "Model 1", "Model 2", "Model 3"),
                       method=c("Movie + user effects", 
                                "Movie + user + genre + year effects", 
                                "Movie effect regularization", 
                                "All effects regularization"),
                       RMSE = c(round(baseline_rmse_f,5),
                                round(mymodel_1_rmse_f,5),
                                round(mymodel_2_rmse_f,5),
                                round(mymodel_3_rmse_f,5)))
rmse_results %>% knitr::kable() # results in a table for the report


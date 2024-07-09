library(tidyverse)
library(caret)
library(ggplot2)
library(dplyr)
library(lubridate)
library(treemapify)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Downloading data files
dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

# Split the ratings variable in userID, movieID, rating, and timestamp, and assign them to the right class.
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

# Splitting the data into a training and test set
# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# transforming timestamp to date
edx <- edx %>%
  mutate(date = as_datetime(timestamp))

# Cleaning up the work space to free memory
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Exploratory data analysis
# distinct movies and users
n_distinct(edx$movieId)
n_distinct(edx$userId)

# Number of movies per genre
genres = edx$genres
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

# Treemap of all genres
# Split genres into individual genres
genres_split <- edx %>%
  separate_rows(genres, sep = "\\|")
# Number of movies per genre
genre_count <- genres_split %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
# Creating a treemap
ggplot(genre_count, aes(area = count, fill = genres, label = genres)) +
  geom_treemap() +
  geom_treemap_text(colour = "white", place = "centre", grow = TRUE) +
  labs(title = "Treemap of Movie Genres", fill = "Genre") +
  theme_minimal()

# Most rated movie
edx %>%
  group_by(title) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  slice(1)

# Top ten best rated movies with at least 10,000 ratings
edx %>%
  group_by(title) %>%
  summarize(average_rating = mean(rating), n = n()) %>%
  filter(n >= 10000) %>%
  arrange(desc(average_rating)) %>%
  top_n(10, average_rating)

# Top ten worst rated movies with at least 10,000 ratings
edx %>%
  group_by(title) %>%
  summarize(average_rating = mean(rating), n = n()) %>%
  filter(n >= 10000) %>%
  arrange(average_rating) %>%
  top_n(-10, average_rating)

# Distribution of ratings
edx %>% ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, fill = "blue", color = "black") +
  labs(title = "Distribution of Ratings", x = "Rating", y = "Count")

# Number of ratings per user
edx %>% 
  count(userId) %>%
  ggplot(aes(x = n)) +
  geom_bar(color = "black") +
  labs(title = "Number of Ratings per User", x = "Number of Ratings", y = "Count") +
  scale_x_continuous(trans = "log10")

# Number of ratings per movie
edx %>% 
  count(movieId) %>%
  ggplot(aes(x = n)) +
  geom_bar(color = "black") +
  labs(title = "Number of Ratings per Movie", x = "Number of Ratings", y = "Count") +
  scale_x_continuous(trans = "log10")


# Data handling

## Missing values

#Checking for missing values in the entire data set
sum(is.na(edx))

#Checking for missing values in individual columns
colSums(is.na(edx))


# Model development

### Different models will be built and evaluated. 

## Baseline model

### We start by developing a baseline model using the average rating.

# Calculating average rating
mu <- mean(edx$rating)

mu

# Use average in baseline model
baseline_model <- rep(mu, nrow(final_holdout_test))

# Calculate RMSE
RMSE_baseline <- RMSE(final_holdout_test$rating, baseline_model)

RMSE_baseline


## Regularization

# Create different values for Lambda
Lambdas <- seq(0, 15, 0.2)

# Calculate RMSE for each Lambda
rmse <- sapply(Lambdas, function(Lambda){
  
  # calculating the mean
  mu <- mean(edx$rating)
  
  # calculating movie effect
  a_i <- edx %>% 
    group_by(movieId) %>%
    summarize(a_i = sum(rating - mu)/(n()+Lambda))
  
  # calculating user effect
  b_u <- edx %>% 
    left_join(a_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - a_i - mu)/(n()+Lambda))
  
  # calculating modelled ratings
  modelled_ratings <- edx %>% 
    left_join(a_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred_rating = mu + a_i + b_u) %>%
    pull(pred_rating)
  
  # return RMSE
  return(RMSE(modelled_ratings, edx$rating))
  
})

# Plot Lambdas vs RMSEs
plot(Lambdas, rmse)

# Which Lambda gives the lowest RMSE?
best_Lambda <- Lambdas[which.min(rmse)]



## Cross validation


# Results and evaluation

## Table of different RMSEs



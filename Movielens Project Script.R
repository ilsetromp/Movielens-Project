library(tidyverse)
library(caret)
library(ggplot2)
library(dplyr)
library(lubridate)
library(treemapify)
library(reshape2)
library(missMDA)
library(Matrix)


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

## Baseline model

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
plot(Lambdas, rmse, type = "b", col = "blue", pch = 19, xlab = "Lambda", ylab = "RMSE",
     main = "RMSE vs Lambda for Regularized Bias")

points(best_Lambda, best_RMSE, col = "red", pch = 19, cex = 2)
text(best_Lambda, best_RMSE, labels = paste0("Lambda=", best_Lambda, "\nRMSE=", 
round(best_RMSE, 4)), pos = 3, col = "red", offset = 1, adj = c(1, 1))


# Which Lambda gives the lowest RMSE?
best_Lambda <- Lambdas[which.min(rmse)]
best_RMSE <- min(rmse)

print(paste("Best Lambda:", best_Lambda))
print(paste("Best RMSE:", best_RMSE))

## Matrix factorization

# Filter out users with fewer than 100 ratings and movies with fewer than 100 ratings
min_ratings <- 100
filtered_edx <- edx %>%
  group_by(userId) %>%
  filter(n() >= min_ratings) %>%
  ungroup() %>%
  group_by(movieId) %>%
  filter(n() >= min_ratings) %>%
  ungroup()

# Create a user-item matrix
y <- filtered_edx %>%
  select(userId, movieId, rating) %>%
  pivot_wider(names_from = movieId, values_from = rating) %>%
  column_to_rownames("userId") %>%
  as.matrix()

lambda <- 0.1  # Set regularization parameter
imputed <- imputePCA(y, ncp = 2, coeff.ridge = lambda)

# Perform SVD on the imputed complete data
svd_res <- svd(imputed$completeObs)

# Get the predicted values
pred_svd <- svd_res$u %*% diag(svd_res$d) %*% t(svd_res$v)

# Prepare the test data
y_test <- final_holdout_test %>%
  select(userId, movieId, rating) %>%
  pivot_wider(names_from = movieId, values_from = rating) %>%
  column_to_rownames("userId") %>%
  as.matrix()

# Define a function to clamp the predictions within the range of valid ratings
clamp <- function(x, lower = 0.5, upper = 5) {
  pmin(pmax(x, lower), upper)
}

# Clamp the predicted values
pred_svd_clamped <- clamp(pred_svd)

# Calculate RMSE for the baseline model
mu <- mean(filtered_edx$rating)
baseline_pred <- matrix(mu, nrow = nrow(y), ncol = ncol(y))
rownames(baseline_pred) <- rownames(y)
colnames(baseline_pred) <- colnames(y)
rmse_baseline <- RMSE(y_test, baseline_pred[rownames(y_test), colnames(y_test)])
print(paste("Baseline RMSE:", rmse_baseline))

# Calculate RMSE for the SVD model
rmse_svd <- RMSE(y_test, pred_svd_clamped[rownames(y_test), colnames(y_test)])
print(paste("Matrix Factorization RMSE (SVD):", rmse_svd))



library(recosystem)

# Prepare the edx data for recosystem
edx_ratings <- edx %>% select(userId, movieId, rating)
test_ratings <- final_holdout_test %>% select(userId, movieId, rating)

# Save the data to temporary files
write.table(edx_ratings, file = "train_data.txt", sep = " ", row.names = FALSE, col.names = FALSE)
write.table(test_ratings, file = "test_data.txt", sep = " ", row.names = FALSE, col.names = FALSE)

# Initialize the recommender model
reco <- Reco()

# Load the training data
train_data <- data_file("train_data.txt")

# Train the model using ALS
reco$train(train_data, opts = list(dim = 20, lrate = 0.1, costp_l2 = 0.01, costq_l2 = 0.01, niter = 20))

# Load the test data
test_data <- data_file("test_data.txt")

# Make predictions
predictions <- tempfile()
reco$predict(test_data, out_file(predictions))

# Load the predictions
predicted_ratings <- scan(predictions)

# Combine the actual and predicted ratings
predicted_ratings_df <- cbind(test_ratings, predicted_rating = predicted_ratings)

# Calculate RMSE for the matrix factorization model
RMSE_matrix_factorization <- RMSE(predicted_ratings_df$rating, predicted_ratings_df$predicted_rating)

# Print the RMSE
print(paste("Matrix Factorization RMSE:", RMSE_matrix_factorization))


### Cross validation

# Results and evaluation

## Table of different RMSEs



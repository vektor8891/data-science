reviews = read.csv("tripadvisor_hotel_reviews.csv", stringsAsFactors=FALSE)

str(reviews)

table(reviews$Rating)


library(tm)
library(SnowballC)


corpus = VCorpus(VectorSource(reviews$Review))

# Look at corpus
corpus[[1]]$content

# Convert to lower-case
corpus = tm_map(corpus, content_transformer(tolower))

corpus[[1]]$content

# Remove punctuation

corpus = tm_map(corpus, removePunctuation)

corpus[[1]]$content

# Remove stopwords and hotel

corpus = tm_map(corpus, removeWords, c("hotel", stopwords("english")))

corpus[[1]]$content

corpus[[1]]$content
# Stem document 

corpus = tm_map(corpus, stemDocument)

corpus[[1]]$content




# Video 6

# Create matrix

frequencies = DocumentTermMatrix(corpus)

frequencies


# Remove sparse terms
sparse = removeSparseTerms(frequencies, 0.95)
sparse

# Convert to a data frame

reviewSparse = as.data.frame(as.matrix(sparse))

# Make all variable names R-friendly

colnames(reviewSparse) = make.names(colnames(reviewSparse))

# Add dependent variable

reviewSparse$UserRating = reviews$Rating

# Split the data

library(caTools)

set.seed(1)

split = sample.split(reviewSparse$UserRating, SplitRatio = 0.7)

trainSparse = subset(reviewSparse, split==TRUE)
testSparse = subset(reviewSparse, split==FALSE)



# Video 7

# Build a CART model

library(rpart)
library(rpart.plot)

reviewCART = rpart(UserRating ~ ., data=trainSparse, method="class")

prp(reviewCART)

# Evaluate the performance of the model
predictCART = predict(reviewCART, newdata=testSparse, type="class")

confusionMatrix = table(testSparse$UserRating, predictCART)
confusionMatrix
# Compute accuracy
sum(diag(confusionMatrix)) / nrow(testSparse)

# Baseline accuracy 

max(table(reviews$Rating)) / nrow(reviews)


# Random forest model

library(randomForest)
set.seed(1)

reviewRF = randomForest(UserRating ~ ., data=trainSparse)

# Make predictions:
predictRF = predict(reviewRF, newdata=testSparse)

confusionMatrixRF = table(testSparse$Rating, predictRF)
confusionMatrixRF
# Compute accuracy
sum(diag(confusionMatrixRF)) / nrow(testSparse)

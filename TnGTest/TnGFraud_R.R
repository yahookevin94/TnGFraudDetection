library(dplyr) # for data manipulation
library(stringr) # for data manipulation
library(caret) # for sampling
library(caTools) # for train/test split
library(ggplot2) # for data visualization
library(corrplot) # for correlations
library(Rtsne) # for tsne plotting
library(performanceEstimation) # for smote implementation
library(ROSE)# for ROSE sampling
library(rpart)# for decision tree model
library(Rborist)# for random forest model
library(xgboost) # for xgboost model

# function to set plot height and width
fig <- function(width, heigth){
  options(repr.plot.width = width, repr.plot.height = heigth)
}

##Removing NA's from dataset##
sample_data <- sample_data[!is.na(sample_data$Class),]

##Summary of Fraud vs Non-Fraud cases##
#0.296% Fraud occurence
sum(sample_data$Class == 1)/(sum(sample_data$Class == 0) + sum(sample_data$Class == 1)) * 100
sum(sample_data$Class == 0)/(sum(sample_data$Class == 0) + sum(sample_data$Class == 1)) * 100

df <- sample_data

##Visualization##
#Class Label Bar Chart#
fig(12, 8)
common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggplot(data = df, aes(x = factor(Class), 
                      y = prop.table(stat(count)), fill = factor(Class),
                      label = scales::percent(prop.table(stat(count))))) +
  geom_bar(position = "dodge") + 
  geom_text(stat = 'count',
            position = position_dodge(.9), 
            vjust = -0.5, 
            size = 3) + 
  scale_x_discrete(labels = c("no fraud", "fraud"))+
  scale_y_continuous(labels = scales::percent)+
  labs(x = 'Class', y = 'Percentage') +
  ggtitle("Distribution of class labels") +
  common_theme

#Time of Transaction by Class#
fig(14, 8)
df %>%
  ggplot(aes(x = Time, fill = factor(Class))) + geom_histogram(bins = 100)+
  labs(x = 'Time in seconds since first transaction', y = 'No. of transactions') +
  ggtitle('Distribution of time of transaction by class') +
  facet_grid(Class ~ ., scales = 'free_y') + common_theme

#Transaction amount by Class#
fig(14, 8)
ggplot(df, aes(x = factor(Class), y = Amount)) + geom_boxplot() + 
  labs(x = 'Class', y = 'Amount') +
  ggtitle("Distribution of transaction amount by class") + common_theme

#Correlation Graph#
fig(14, 8)
correlations <- cor(df[,-1],method="pearson")
corrplot(correlations, number.cex = .9, method = "circle", type = "full", tl.cex=0.8,tl.col = "black")

##t-Distributed Stochastic Neighbour Embedding, dimensionality reduction##
fig(16, 10)

# Use 80% of data to compute t-SNE
tsne_subset <- 1:as.integer(0.8*nrow(df))
tsne <- Rtsne(df[tsne_subset,-c(1, 31)], perplexity = 20, theta = 0.5, pca = F, verbose = F, max_iter = 500, check_duplicates = F)

classes <- as.factor(df$Class[tsne_subset])
tsne_mat <- as.data.frame(tsne$Y)
ggplot(tsne_mat, aes(x = V1, y = V2)) + geom_point(aes(color = classes)) + theme_minimal() + common_theme + ggtitle("t-SNE visualisation of transactions") + scale_color_manual(values = c("#E69F00", "#56B4E9"))

##Dealing with Imbalanced Dataset##
#Remove Time variable (Only used to chronologically list variables)#
df <- df[,-1]

#Change 'Class' variable to factor
df$Class <- as.factor(df$Class)
levels(df$Class) <- c("Not_Fraud", "Fraud")

#Scale numeric variables
df[,-30] <- scale(df[,-30])

head(df)

##Splitting Dataset##
set.seed(321)
split <- sample.split(df$Class, SplitRatio = 0.7)
train <-  subset(df, split == TRUE)
test <- subset(df, split == FALSE)

##Sample Method comparison##
# downsampling
set.seed(123)
down_train <- downSample(x = train[, -ncol(train)],
                         y = train$Class)
table(down_train$Class)

# upsampling
set.seed(123)
up_train <- upSample(x = train[, -ncol(train)],
                     y = train$Class)
table(up_train$Class)

# smote
set.seed(123)
smote_train <- smote(Class ~ ., data  = train)

table(smote_train$Class)

# rose
set.seed(123)
rose_train <- ROSE(Class ~ ., data  = train)$data 

table(rose_train$Class)

##Performance Evaluation for Sample Methods##
#CART Model Performance on imbalanced data
set.seed(123)

orig_fit <- rpart(Class ~ ., data = train)

#Evaluate model performance on test set
pred_orig <- predict(orig_fit, newdata = test, method = "class")

roc.curve(test$Class, pred_orig[,2], plotit = TRUE)

#Decision Tree test on Sampling Methods#
set.seed(123)
# Build down-sampled model
down_fit <- rpart(Class ~ ., data = down_train)

set.seed(123)
# Build up-sampled model
up_fit <- rpart(Class ~ ., data = up_train)

set.seed(123)
# Build smote model
smote_fit <- rpart(Class ~ ., data = smote_train)

set.seed(123)
# Build rose model
rose_fit <- rpart(Class ~ ., data = rose_train)

##AUC Comparison of D.Tree Models##
# AUC on down-sampled data
pred_down <- predict(down_fit, newdata = test)

print('Fitting model to downsampled data')
roc.curve(test$Class, pred_down[,2], plotit = FALSE)

# AUC on up-sampled data
pred_up <- predict(up_fit, newdata = test)

print('Fitting model to upsampled data')
roc.curve(test$Class, pred_up[,2], plotit = FALSE)

# AUC on SMOTE data
pred_smote <- predict(smote_fit, newdata = test)

print('Fitting model to smote data')
roc.curve(test$Class, pred_smote[,2], plotit = FALSE)

# AUC on ROSE data
pred_rose <- predict(rose_fit, newdata = test)

print('Fitting model to rose data')
roc.curve(test$Class, pred_rose[,2], plotit = FALSE)

##Models on Upsampled Data (SMOTE)##
#Logistic Regression#
glm_fit <- glm(Class ~ ., data = smote_train, family = 'binomial')

pred_glm <- predict(glm_fit, newdata = test, type = 'response')

roc.curve(test$Class, pred_glm, plotit = TRUE)

#Random Forest#
x = smote_train[, -30]
y = smote_train[,30]

rf_fit <- Rborist(x, y, ntree = 1000, minNode = 20, maxLeaf = 13)

rf_pred <- predict(rf_fit, test[,-30], ctgCensus = "prob")
prob <- rf_pred$prob

roc.curve(test$Class, prob[,2], plotit = TRUE)

#XGBoost#
# Convert class labels from factor to numeric

labels <- smote_train$Class

y <- recode(labels, 'Not_Fraud' = 0, "Fraud" = 1)

set.seed(123)
xgb <- xgboost(data = data.matrix(smote_train[,-30]), 
               label = y,
               eta = 0.1,
               gamma = 0.1,
               max_depth = 10, 
               nrounds = 300, 
               objective = "binary:logistic",
               colsample_bytree = 0.6,
               verbose = 0,
               nthread = 7,
)

xgb_pred <- predict(xgb, data.matrix(test[,-30]))

roc.curve(test$Class, xgb_pred, plotit = TRUE)

#XGBoost Confusion Matrix
xgb_pred <-  as.numeric(xgb_pred > 0.98) #Decide probability value based on requirements

table(recode(test$Class, 'Not_Fraud' = 0, "Fraud" = 1), xgb_pred)

##Feature Importance##
names <- dimnames(data.matrix(smote_train[,-30]))[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgb)
# Nice graph
xgb.plot.importance(importance_matrix[1:10,])


##Conclusion: SMOTE Sampling Method used in tangent with XGBoost Machine Learning Model yielded the best results in terms of AUC Scores: 0.994##

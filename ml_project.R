library(AppliedPredictiveModeling)
library(caret)
library(lattice)
library(randomforest)
library(sqldf)
library(gbm)

training <- read.csv("E:/Coursera/machine_learning/pml-training.csv")
set.seed(1337)

# Pre-Processing

# Delete columns that have any NA values
training <- training[, !is.na(colSums( training == 'NA'))]

# Separate out 25% of the data for validation purposes at the end
training_partition = createDataPartition(training$classe, p = 0.75)[[1]]
training_set = training[training_partition,]
testing_set = training[-training_partition,]

# 5 fold cross validation control
control <- trainControl(method = "repeatedcv", number = 5, repeats = 2)

# For this model, I've chosen to use a quick and dirty select of all values that are not
# averages, maxes, mins, or standard deviations. This uses the boosted tree model "gbm" package
train_model1 <- train(training_set$classe ~ 
                        user_name	+
                        roll_belt	+
                        pitch_belt	+
                        yaw_belt	+
                        total_accel_belt	+
                        gyros_belt_x	+
                        gyros_belt_y	+
                        gyros_belt_z	+
                        accel_belt_x	+
                        accel_belt_y	+
                        accel_belt_z	+
                        magnet_belt_x	+
                        magnet_belt_y	+
                        magnet_belt_z	+
                        roll_arm	+
                        pitch_arm	+
                        yaw_arm	+
                        total_accel_arm	+
                        gyros_arm_x	+
                        gyros_arm_y	+
                        gyros_arm_z	+
                        accel_arm_x	+
                        accel_arm_y	+
                        accel_arm_z	+
                        magnet_arm_x	+
                        magnet_arm_y	+
                        magnet_arm_z	+
                        roll_dumbbell	+
                        pitch_dumbbell	+
                        yaw_dumbbell	+
                        total_accel_dumbbell	+
                        gyros_dumbbell_x	+
                        gyros_dumbbell_y	+
                        gyros_dumbbell_z	+
                        accel_dumbbell_x	+
                        accel_dumbbell_y	+
                        accel_dumbbell_z	+
                        magnet_dumbbell_x	+
                        magnet_dumbbell_y	+
                        magnet_dumbbell_z	+
                        roll_forearm	+
                        pitch_forearm	+
                        yaw_forearm	+
                        total_accel_forearm	+
                        gyros_forearm_x	+
                        gyros_forearm_y	+
                        gyros_forearm_z	+
                        accel_forearm_x	+
                        accel_forearm_y	+
                        accel_forearm_z	+
                        magnet_forearm_x	+
                        magnet_forearm_y	+
                        magnet_forearm_z
                      ,data = training_set, trControl = control, method = "gbm")

# Confusion matrix using the testing_set created initially with 25% of the test data
confusionMatrix(predict(train_model1, testing_set), testing_set$classe)

# Predict the 20 classifications from the pml-testing file
testing <- read.csv("E:/Coursera/machine_learning/pml-testing.csv")
answers = predict(train_model1, testing)


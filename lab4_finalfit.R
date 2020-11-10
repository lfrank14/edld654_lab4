library(tidyverse)
library(tidymodels)
set.seed(3000)


## Prepare for parallel processing
all_cores <- parallel::detectCores(logical = FALSE)

library(doParallel)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})


## Load and prep training data
df_train <- read_csv("data/train.csv") %>% 
  mutate(classification_binary = ifelse(classification < 3, "below", "proficient"))
# df_train <- df_train %>%
#   sample_frac(size = .005)


## Create initial splits & folds
splits <- initial_split(df_train)
train <- training(splits)
test <- testing(splits)
cv_splits <- vfold_cv(train)


## Create basic recipe
rec <- recipe(classification_binary ~ gndr + ethnic_cd + lat + lon, df_train) %>% 
  step_unknown(all_nominal(), -all_outcomes()) %>% 
  step_nzv(all_predictors(), freq_cut = 0, unique_cut = 0) %>% 
  step_medianimpute(all_numeric()) %>% 
  step_center(all_numeric()) %>% 
  step_scale(all_numeric()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_nzv(all_predictors())


## Define KNN model
knn_mod_tune <- nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("classification") %>% 
  set_args(neighbors = tune(),
           dist_power = tune())


## Load tuned model fits
knn_fit_tune <- readRDS("models/knn_fit_tune.Rds")


## Save best metrics and finalize model
knn_best <- knn_fit_tune %>% 
  select_best(metric = "roc_auc")

knn_mod_final <- knn_mod_tune %>% 
  finalize_model(knn_best)

rec_final <- rec %>% 
  finalize_recipe(knn_best)


## Run final model
knn_fit_final <- last_fit(
  knn_mod_final,
  preprocessor = rec_final,
  split = splits)
saveRDS(knn_fit_final, "models/knn_fit_final.Rds")

knn_fit_final %>% 
  collect_metrics()

knn_fit_final %>% 
  collect_predictions() %>% 
  conf_mat(truth = classification_binary, estimate = .pred_class)

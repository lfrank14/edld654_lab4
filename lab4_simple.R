library(tidyverse)
library(tidymodels)

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
prep(rec)


## Create and run simple KNN model
knn_mod <- nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("classification") %>% 
  set_args(neighbors = 5)

knn_fit_simple <- fit_resamples(knn_mod,
                         rec,
                         cv_splits,
                         control = tune::control_resamples(save_pred = TRUE))
saveRDS(knn_fit_simple, "models/knn_fit_simple.Rds")


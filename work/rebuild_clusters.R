workspace_dir <- normalizePath(".", winslash = "/", mustWork = TRUE)
local_library <- file.path(workspace_dir, "work", "Rlibs")
.libPaths(c(local_library, .libPaths()))

library(clustrd)

Sys.setlocale("LC_CTYPE", "Korean_Korea.949")
Sys.setlocale("LC_COLLATE", "Korean_Korea.949")

input_path <- file.path(
  workspace_dir,
  "work",
  "plot_data_rebuild",
  "KAIA_DA_k=6_model_b_input.csv"
)
output_path <- file.path(
  workspace_dir,
  "work",
  "plot_data_rebuild",
  "KAIA_DA_k=6_model_b_cluster_result.csv"
)
summary_path <- file.path(
  workspace_dir,
  "work",
  "plot_data_rebuild",
  "KAIA_DA_k=6_model_b_cluster_run_summary.csv"
)

model_df <- read.csv(
  input_path,
  fileEncoding = "UTF-8-BOM",
  stringsAsFactors = FALSE,
  check.names = FALSE
)
case_id <- model_df$case_id
model_active <- model_df[, setdiff(names(model_df), "case_id"), drop = FALSE]
model_active[] <- lapply(model_active, as.factor)

fixed_asw <- tuneclus(
  model_active,
  nclusrange = 6,
  ndimrange = 5,
  method = "clusCA",
  criterion = "asw",
  nstart = 100,
  seed = 1234
)

cluster_result <- data.frame(
  case_id = case_id,
  cluster = as.integer(fixed_asw$clusobjbest$cluster)
)
write.csv(cluster_result, output_path, row.names = FALSE, fileEncoding = "UTF-8")

cluster_sizes <- as.data.frame(table(cluster_result$cluster), stringsAsFactors = FALSE)
names(cluster_sizes) <- c("cluster", "count")
cluster_sizes$mean_asw <- fixed_asw$critbest
write.csv(cluster_sizes, summary_path, row.names = FALSE, fileEncoding = "UTF-8")

print(fixed_asw)
print(cluster_sizes)
cat("cluster_output=", output_path, "\n", sep = "")

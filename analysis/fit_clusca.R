root_dir <- normalizePath(".", winslash = "/", mustWork = TRUE)
input_path <- file.path(root_dir, "data", "processed", "model_b_input.csv")
output_path <- file.path(root_dir, "data", "processed", "cluster_result.csv")
summary_path <- file.path(root_dir, "results", "clusca_run_summary.csv")

if (!requireNamespace("clustrd", quietly = TRUE)) {
  stop("R package 'clustrd' is required. Run install.packages('clustrd').")
}

model_df <- read.csv(
  input_path,
  fileEncoding = "UTF-8-BOM",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

if (nrow(model_df) != 739L || !"case_id" %in% names(model_df)) {
  stop("The processed Model B input must contain case_id and 739 observations.")
}

case_id <- model_df$case_id
active <- model_df[, setdiff(names(model_df), "case_id"), drop = FALSE]
active[] <- lapply(active, as.factor)

fit <- clustrd::tuneclus(
  active,
  nclusrange = 6,
  ndimrange = 5,
  method = "clusCA",
  criterion = "asw",
  nstart = 100,
  seed = 1234
)

cluster_result <- data.frame(
  case_id = case_id,
  cluster = as.integer(fit$clusobjbest$cluster)
)
cluster_sizes <- as.data.frame(table(cluster_result$cluster), stringsAsFactors = FALSE)
names(cluster_sizes) <- c("cluster", "count")
cluster_sizes$mean_asw <- fit$critbest

dir.create(dirname(summary_path), recursive = TRUE, showWarnings = FALSE)
write.csv(cluster_result, output_path, row.names = FALSE, fileEncoding = "UTF-8")
write.csv(cluster_sizes, summary_path, row.names = FALSE, fileEncoding = "UTF-8")

print(cluster_sizes)


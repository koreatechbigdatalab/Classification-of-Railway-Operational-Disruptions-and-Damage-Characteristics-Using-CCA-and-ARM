library(clustrd)

base_dir <- "D:/OneDrive/바탕 화면/Yoonity_KAIA"
save_dir <- file.path(base_dir, "outputs")
version_name <- "KAIA_DA_k=6"
selected_model <- "model_b"
selected_k <- 6
selected_d <- 5
nstart_value <- 100
seed_value <- 1234

input_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_input.csv")
)

if (!file.exists(input_path)) {
  stop(paste("입력 파일을 찾을 수 없습니다:", input_path))
}

model_df <- read.csv(
  input_path,
  fileEncoding = "UTF-8-BOM",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

case_id <- model_df$case_id

model_active <- model_df[
  ,
  setdiff(names(model_df), "case_id"),
  drop = FALSE
]

model_active[] <- lapply(model_active, as.factor)

cat("\n============================================================\n")
cat("분석 버전:", version_name, "\n")
cat("모형:", selected_model, "\n")
cat("고정 군집 수:", selected_k, "\n")
cat("고정 차원 수:", selected_d, "\n")
cat("입력 파일:", input_path, "\n")
cat("사례 수:", nrow(model_df), "\n")
cat("적극변수 수:", ncol(model_active), "\n")
cat("적극변수:", paste(names(model_active), collapse = ", "), "\n")
cat("============================================================\n")

# 3~7개 전체 탐색에서 선택된 최적해가 아니라,
# 비교를 위해 지정한 6군집·5차원 해를 동일 조건으로 적합한다.
fixed_asw <- tuneclus(
  model_active,
  nclusrange = selected_k,
  ndimrange = selected_d,
  method = "clusCA",
  criterion = "asw",
  nstart = nstart_value,
  seed = seed_value
)

fixed_ch <- tuneclus(
  model_active,
  nclusrange = selected_k,
  ndimrange = selected_d,
  method = "clusCA",
  criterion = "ch",
  nstart = nstart_value,
  seed = seed_value
)

cat("\n[고정 6군집·5차원 ASW 결과]\n")
print(fixed_asw)

cat("\n[고정 6군집·5차원 CH 결과]\n")
print(fixed_ch)

stopifnot(
  fixed_asw$nclusbest == selected_k,
  fixed_asw$ndimbest == selected_d,
  fixed_ch$nclusbest == selected_k,
  fixed_ch$ndimbest == selected_d,
  identical(
    as.integer(fixed_asw$clusobjbest$cluster),
    as.integer(fixed_ch$clusobjbest$cluster)
  )
)

cluster_result_asw <- data.frame(
  case_id = case_id,
  cluster = fixed_asw$clusobjbest$cluster
)

cluster_result_ch <- data.frame(
  case_id = case_id,
  cluster = fixed_ch$clusobjbest$cluster
)

cluster_output_path_asw <- file.path(
  save_dir,
  paste0(
    version_name,
    "_",
    selected_model,
    "_cluster_result_asw.csv"
  )
)

cluster_output_path_ch <- file.path(
  save_dir,
  paste0(
    version_name,
    "_",
    selected_model,
    "_cluster_result_ch.csv"
  )
)

write.csv(
  cluster_result_asw,
  cluster_output_path_asw,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

write.csv(
  cluster_result_ch,
  cluster_output_path_ch,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

fixed_summary <- data.frame(
  version = version_name,
  model = selected_model,
  n = nrow(model_df),
  n_variables = ncol(model_active),
  selected_k = selected_k,
  selected_d = selected_d,
  asw_value = fixed_asw$critbest,
  ch_value = fixed_ch$critbest,
  cluster_output_path_asw = cluster_output_path_asw,
  cluster_output_path_ch = cluster_output_path_ch,
  stringsAsFactors = FALSE
)

fixed_summary_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_fixed_summary.csv")
)

write.csv(
  fixed_summary,
  fixed_summary_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

results_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_cca_results.rds")
)

saveRDS(
  list(
    version = version_name,
    model = selected_model,
    data = model_df,
    active_data = model_active,
    selected_k = selected_k,
    selected_d = selected_d,
    fixed_asw = fixed_asw,
    fixed_ch = fixed_ch,
    cluster_result_asw = cluster_result_asw,
    cluster_result_ch = cluster_result_ch,
    summary = fixed_summary
  ),
  results_path
)

asw_cluster_summary <- data.frame(
  solution = "fixed_k6_d5",
  mean_asw = fixed_asw$critbest,
  cluster = seq_along(fixed_asw$cluasw),
  cluster_asw = as.numeric(fixed_asw$cluasw)
)

asw_summary_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_asw_summary.csv")
)

write.csv(
  asw_cluster_summary,
  asw_summary_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

cat("\n[고정 6군집·5차원 요약]\n")
print(fixed_summary)
cat("\n[평균 및 군집별 ASW]\n")
print(asw_cluster_summary)

# ============================================================
# Model B 고정 6군집·5차원 해의 bootstrap Jaccard 안정성
# ============================================================

jaccard_nboot <- 100
jaccard_nstart <- 100

cat("\n============================================================\n")
cat("Jaccard 안정성 분석: 고정 6군집·5차원 해\n")
cat("k =", selected_k, "\n")
cat("d =", selected_d, "\n")
cat("nboot =", jaccard_nboot, "\n")
cat("nstart =", jaccard_nstart, "\n")
cat("============================================================\n")

jaccard_result <- local_bootclus(
  data = model_active,
  nclus = selected_k,
  ndim = selected_d,
  method = "clusCA",
  nstart = jaccard_nstart,
  nboot = jaccard_nboot,
  seed = seed_value
)

jaccard_matrix <- as.matrix(jaccard_result$Jaccard)

stopifnot(
  nrow(jaccard_matrix) == jaccard_nboot,
  ncol(jaccard_matrix) == selected_k
)

jaccard_summary <- data.frame(
  solution = "fixed_k6_d5",
  cluster = seq_len(ncol(jaccard_matrix)),
  mean_jaccard = round(colMeans(jaccard_matrix, na.rm = TRUE), 3),
  median_jaccard = round(
    apply(jaccard_matrix, 2, median, na.rm = TRUE),
    3
  ),
  q1_jaccard = round(
    apply(jaccard_matrix, 2, quantile, probs = 0.25, na.rm = TRUE),
    3
  ),
  q3_jaccard = round(
    apply(jaccard_matrix, 2, quantile, probs = 0.75, na.rm = TRUE),
    3
  ),
  min_jaccard = round(
    apply(jaccard_matrix, 2, min, na.rm = TRUE),
    3
  ),
  max_jaccard = round(
    apply(jaccard_matrix, 2, max, na.rm = TRUE),
    3
  )
)

jaccard_summary$stability <- ifelse(
  jaccard_summary$mean_jaccard >= 0.85,
  "매우 안정적",
  ifelse(
    jaccard_summary$mean_jaccard >= 0.75,
    "안정적",
    ifelse(
      jaccard_summary$mean_jaccard >= 0.60,
      "경계적",
      "불안정"
    )
  )
)

jaccard_long <- do.call(
  rbind,
  lapply(seq_len(ncol(jaccard_matrix)), function(cluster_id) {
    data.frame(
      solution = "fixed_k6_d5",
      bootstrap = seq_len(nrow(jaccard_matrix)),
      cluster = cluster_id,
      jaccard = jaccard_matrix[, cluster_id]
    )
  })
)

jaccard_summary_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_jaccard_summary.csv")
)

jaccard_bootstrap_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_jaccard_bootstrap.csv")
)

jaccard_results_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_jaccard_results.rds")
)

write.csv(
  jaccard_summary,
  jaccard_summary_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

write.csv(
  jaccard_long,
  jaccard_bootstrap_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

saveRDS(
  jaccard_result,
  jaccard_results_path
)

cat("\n[bootstrap Jaccard 안정성 요약]\n")
print(jaccard_summary)
cat("고정해 요약 저장:", fixed_summary_path, "\n")
cat("군집 결과 저장(ASW):", cluster_output_path_asw, "\n")
cat("군집 결과 저장(CH):", cluster_output_path_ch, "\n")
cat("R 결과 객체 저장:", results_path, "\n")
cat("Jaccard 요약 저장:", jaccard_summary_path, "\n")
cat("Jaccard 반복 결과 저장:", jaccard_bootstrap_path, "\n")
cat("Jaccard R 객체 저장:", jaccard_results_path, "\n")
# ============================================================
# K=6 군집과 범주형 변수의 Haberman 조정 표준화 잔차(ASR)
# ============================================================

asr_threshold <- 1.96
top_n_per_variable <- 2

case_lookup_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_case_lookup.csv")
)

case_lookup_df <- read.csv(
  case_lookup_path,
  fileEncoding = "UTF-8-BOM",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

profile_df <- merge(
  cluster_result_asw,
  model_df,
  by = "case_id",
  all.x = TRUE,
  sort = FALSE
)

if ("근본원인별 그룹" %in% names(case_lookup_df)) {
  profile_df <- merge(
    profile_df,
    case_lookup_df[, c("case_id", "근본원인별 그룹"), drop = FALSE],
    by = "case_id",
    all.x = TRUE,
    sort = FALSE
  )
}

stopifnot(
  nrow(profile_df) == nrow(model_df),
  !anyNA(profile_df$cluster)
)

profile_variables <- setdiff(names(model_df), "case_id")
if ("근본원인별 그룹" %in% names(profile_df)) {
  profile_variables <- c(profile_variables, "근본원인별 그룹")
}

calculate_asr <- function(data, variable_name) {
  cluster_values <- factor(
    data$cluster,
    levels = sort(unique(data$cluster))
  )
  category_values <- as.character(data[[variable_name]])
  category_values[is.na(category_values)] <- "결측"
  category_values <- factor(category_values)

  observed <- table(cluster_values, category_values)
  n_total <- sum(observed)
  row_totals <- rowSums(observed)
  col_totals <- colSums(observed)
  expected <- outer(row_totals, col_totals) / n_total

  denominator <- sqrt(
    expected
    * (1 - row_totals / n_total)
    * rep(1 - col_totals / n_total, each = nrow(observed))
  )
  dim(denominator) <- dim(expected)

  adjusted_residual <- (observed - expected) / denominator

  result <- expand.grid(
    cluster = as.integer(rownames(observed)),
    category = colnames(observed),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  result$variable <- variable_name
  result$observed <- as.numeric(observed)
  result$expected <- as.numeric(expected)
  result$adjusted_residual <- as.numeric(adjusted_residual)
  result$cluster_share_pct <- as.numeric(
    sweep(observed, 1, row_totals, "/") * 100
  )
  result$overall_share_pct <- rep(
    as.numeric(col_totals / n_total * 100),
    each = nrow(observed)
  )
  result$meets_positive_1_96 <- (
    result$adjusted_residual >= asr_threshold
  )
  result$meets_negative_minus_1_96 <- (
    result$adjusted_residual <= -asr_threshold
  )

  chisq_result <- suppressWarnings(chisq.test(observed, correct = FALSE))
  cramer_v <- sqrt(
    unname(chisq_result$statistic)
    / (n_total * min(nrow(observed) - 1, ncol(observed) - 1))
  )

  test_summary <- data.frame(
    variable = variable_name,
    n = n_total,
    rows_clusters = nrow(observed),
    categories = ncol(observed),
    chi_square = unname(chisq_result$statistic),
    degrees_freedom = unname(chisq_result$parameter),
    p_value = chisq_result$p.value,
    cramers_v = cramer_v,
    min_expected = min(chisq_result$expected),
    expected_below_5_cells = sum(chisq_result$expected < 5),
    expected_below_5_pct = mean(chisq_result$expected < 5) * 100,
    stringsAsFactors = FALSE
  )

  list(cells = result, test = test_summary)
}

asr_objects <- lapply(
  profile_variables,
  function(variable_name) calculate_asr(profile_df, variable_name)
)

asr_all_cells <- do.call(
  rbind,
  lapply(asr_objects, function(x) x$cells)
)

asr_chi_square_summary <- do.call(
  rbind,
  lapply(asr_objects, function(x) x$test)
)

positive_cells <- asr_all_cells[
  asr_all_cells$adjusted_residual >= asr_threshold,
  ,
  drop = FALSE
]

positive_cells <- positive_cells[
  order(
    positive_cells$cluster,
    positive_cells$variable,
    -positive_cells$adjusted_residual,
    -positive_cells$observed
  ),
  ,
  drop = FALSE
]

group_key <- interaction(
  positive_cells$cluster,
  positive_cells$variable,
  drop = TRUE,
  lex.order = TRUE
)

asr_selected_top2 <- do.call(
  rbind,
  lapply(
    split(positive_cells, group_key),
    function(group_df) head(group_df, top_n_per_variable)
  )
)

asr_selected_top2$rank_within_cluster_variable <- ave(
  asr_selected_top2$adjusted_residual,
  asr_selected_top2$cluster,
  asr_selected_top2$variable,
  FUN = seq_along
)

asr_selected_top2 <- asr_selected_top2[
  ,
  c(
    "cluster",
    "variable",
    "rank_within_cluster_variable",
    "category",
    "observed",
    "expected",
    "adjusted_residual",
    "cluster_share_pct",
    "overall_share_pct"
  )
]

asr_all_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_ASR_all_cells.csv")
)
asr_selected_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_ASR_selected_top2.csv")
)
asr_chi_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_ASR_chi_square_summary.csv")
)

write.csv(
  asr_all_cells,
  asr_all_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)
write.csv(
  asr_selected_top2,
  asr_selected_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)
write.csv(
  asr_chi_square_summary,
  asr_chi_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

cat("\n[ASR ≥ 1.96, 군집×변수별 상위 2개]\n")
print(asr_selected_top2)
cat("ASR 전체 셀 저장:", asr_all_path, "\n")
cat("ASR 선정 결과 저장:", asr_selected_path, "\n")
cat("카이제곱·Cramer's V 요약 저장:", asr_chi_path, "\n")

#!/usr/bin/env Rscript
# proDA fit for PTERO.  Invoked as:
#   Rscript proda_fit.R <matrix.tsv> <design.tsv> <ref_group> <out.tsv>
#
# matrix.tsv : tab-separated; first column = protein id (row names), remaining
#              columns = samples; values = normalized log2 intensities; missing = NA.
#              proDA models the missing values -- do NOT impute or drop them.
# design.tsv : tab-separated with columns 'sample' and 'group'.
# out.tsv    : written with columns name, diff, t_statistic, se, pval, adj_pval,
#              df, avg_abundance  (diff = log2 fold change, non-reference vs reference).
#
# Prints a final line "PRODA_OK coef=... rver=... pver=..." on success, or a line
# beginning "PRODA_ERROR ..." and a non-zero exit code on failure.

if (!requireNamespace("proDA", quietly = TRUE)) {
  cat("PRODA_ERROR proDA package not installed\n"); quit(status = 3, save = "no")
}
suppressMessages(library(proDA))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) { cat("PRODA_ERROR usage: matrix design ref out\n"); quit(status = 2, save = "no") }
matrix_file <- args[1]; design_file <- args[2]; ref <- args[3]; out_file <- args[4]
seed <- if (length(args) >= 5) suppressWarnings(as.integer(args[5])) else 1L
if (is.na(seed)) seed <- 1L

df <- read.delim(matrix_file, check.names = FALSE, stringsAsFactors = FALSE)
rn <- as.character(df[[1]])
mat <- as.matrix(df[, -1, drop = FALSE])
rownames(mat) <- rn
storage.mode(mat) <- "double"

design <- read.delim(design_file, check.names = FALSE, stringsAsFactors = FALSE)
groups <- design$group[match(colnames(mat), design$sample)]
if (any(is.na(groups))) { cat("PRODA_ERROR design does not cover all matrix columns\n"); quit(status = 4, save = "no") }
if (!(ref %in% groups)) { cat("PRODA_ERROR reference group not present in design\n"); quit(status = 5, save = "no") }

cd <- data.frame(condition = factor(groups))
set.seed(seed)   # proDA initialises the fit with random draws; fix for reproducibility
fit <- proDA(mat, design = ~ condition, col_data = cd, reference_level = ref)
coef <- setdiff(result_names(fit), "Intercept")
if (length(coef) < 1) { cat("PRODA_ERROR no non-intercept coefficient to test\n"); quit(status = 6, save = "no") }
coef <- coef[1]

res <- test_diff(fit, contrast = coef, pval_adjust_method = "BH")
res <- res[, c("name", "diff", "t_statistic", "se", "pval", "adj_pval", "df", "avg_abundance")]
write.table(res, out_file, sep = "\t", row.names = FALSE, quote = FALSE)

cat(sprintf("PRODA_OK coef=%s rver=%s.%s pver=%s seed=%d\n", coef,
            R.version$major, R.version$minor, as.character(packageVersion("proDA")), seed))
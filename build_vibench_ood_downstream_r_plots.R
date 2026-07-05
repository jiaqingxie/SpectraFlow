#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

if (.Platform$OS.type == "windows") {
  windowsFonts(Arial = windowsFont("Arial"))
}

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[[idx + 1]]
}

metrics_csv <- get_arg("--metrics_csv", "results/vibench_ood_downstream_plots/vibench_ood_downstream_metrics.csv")
out_dir <- get_arg("--out_dir", "results/vibench_ood_downstream_plots/r_plots")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "by_dataset_pointrange"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "starplots"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "heatmaps"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "dumbbell"), recursive = TRUE, showWarnings = FALSE)

dataset_order <- c("pahs", "nist_ir", "peptide_mod", "peptide", "geom", "zinc15", "qm9", "mols")
property_order <- c(
  "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
  "RingCount", "FractionCSP3", "NumAromaticRings", "MolMR", "LabuteASA"
)
excluded_properties <- c("MolWt", "HeavyAtomCount")
feature_order <- c("IR raw", "Raman raw", "Flow embedding", "Morgan fingerprint", "Flow + Morgan FP")
feature_short <- c(
  "IR raw" = "IR",
  "Raman raw" = "Raman",
  "Flow embedding" = "Flow",
  "Morgan fingerprint" = "FP",
  "Flow + Morgan FP" = "Flow+FP"
)
feature_colors <- c(
  "IR raw" = "#737373",
  "Raman raw" = "#4C78A8",
  "Flow embedding" = "#F58518",
  "Morgan fingerprint" = "#54A24B",
  "Flow + Morgan FP" = "#B279A2"
)
feature_lty <- c(
  "IR raw" = 1,
  "Raman raw" = 1,
  "Flow embedding" = 1,
  "Morgan fingerprint" = 1,
  "Flow + Morgan FP" = 2
)
feature_pch <- c(
  "IR raw" = NA,
  "Raman raw" = NA,
  "Flow embedding" = NA,
  "Morgan fingerprint" = 21,
  "Flow + Morgan FP" = 22
)
property_labels <- c(
  "LogP" = "LogP",
  "TPSA" = "TPSA",
  "NumHDonors" = "H donors",
  "NumHAcceptors" = "H acceptors",
  "NumRotatableBonds" = "Rot. bonds",
  "RingCount" = "Rings",
  "FractionCSP3" = "Frac. CSP3",
  "NumAromaticRings" = "Arom. rings",
  "MolMR" = "MolMR",
  "LabuteASA" = "LabuteASA"
)

theme_pub <- function(base_size = 10) {
  theme_minimal(base_family = "Arial", base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 4, margin = margin(b = 6)),
      plot.subtitle = element_text(size = base_size + 1, colour = "grey30", margin = margin(b = 8)),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(colour = "grey20"),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      legend.title = element_blank(),
      legend.position = "top",
      strip.text = element_text(face = "bold"),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    )
}

clip_r2 <- function(x) pmax(pmin(x, 1), -1)

df <- read.csv(metrics_csv, stringsAsFactors = FALSE)
df <- df[!(df$property %in% excluded_properties), ]
df$dataset <- factor(df$dataset, levels = dataset_order)
df$property <- factor(df$property, levels = property_order)
df$feature <- factor(df$feature, levels = feature_order)
df$feature_short <- factor(feature_short[as.character(df$feature)], levels = feature_short[feature_order])
df$r2_plot <- clip_r2(df$r2_mean)
df$r2_low <- clip_r2(df$r2_mean - df$r2_std)
df$r2_high <- clip_r2(df$r2_mean + df$r2_std)

# Dataset-level summary: mean across properties, with typical seed std across properties.
summary_df <- aggregate(
  cbind(r2_mean, r2_std) ~ dataset + feature + feature_short,
  data = df,
  FUN = function(x) mean(x, na.rm = TRUE)
)
names(summary_df)[names(summary_df) == "r2_mean"] <- "mean_r2"
names(summary_df)[names(summary_df) == "r2_std"] <- "mean_seed_std"
summary_df$mean_r2_plot <- clip_r2(summary_df$mean_r2)
summary_df$low <- clip_r2(summary_df$mean_r2 - summary_df$mean_seed_std)
summary_df$high <- clip_r2(summary_df$mean_r2 + summary_df$mean_seed_std)
summary_df$dataset <- factor(summary_df$dataset, levels = rev(dataset_order))
summary_df$feature <- factor(summary_df$feature, levels = feature_order)
summary_df$feature_short <- factor(summary_df$feature_short, levels = feature_short[feature_order])

bar_df <- summary_df
bar_df$dataset <- factor(as.character(bar_df$dataset), levels = dataset_order)
bar_df$feature <- factor(bar_df$feature, levels = feature_order)

p_bar_vertical <- ggplot(bar_df, aes(x = mean_r2_plot, y = dataset, fill = feature)) +
  geom_vline(xintercept = 0, linewidth = 0.45, colour = "grey25") +
  geom_col(
    position = position_dodge(width = 0.76),
    width = 0.68,
    colour = "grey20",
    linewidth = 0.18,
    alpha = 0.92
  ) +
  geom_errorbar(
    aes(xmin = low, xmax = high),
    orientation = "y",
    position = position_dodge(width = 0.76),
    width = 0.20,
    linewidth = 0.35,
    colour = "grey20"
  ) +
  scale_fill_manual(values = feature_colors, breaks = feature_order) +
  scale_x_continuous(breaks = seq(-1, 1, 0.5), position = "bottom") +
  coord_cartesian(xlim = c(-1, 1)) +
  labs(
    title = "Average downstream R2 by dataset",
    x = "Mean R2",
    y = NULL
  ) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE)) +
  theme_pub(13) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.background = element_blank(),
    plot.title = element_text(face = "bold", size = 22, margin = margin(b = 10)),
    axis.text.x = element_text(size = 13),
    axis.text.y = element_text(size = 16, angle = 90, hjust = 0.5, vjust = 0.5),
    axis.title = element_text(size = 16, face = "bold"),
    legend.text = element_text(size = 12),
    panel.grid.major.y = element_line(colour = "grey90", linewidth = 0.35),
    plot.margin = margin(8, 8, 8, 6)
  )

ggsave(
  filename = file.path(out_dir, "r_vibench_dataset_feature_average_barplot.svg"),
  plot = p_bar_vertical,
  width = 6.8,
  height = 10.4
)

p_overview <- ggplot(summary_df, aes(x = mean_r2_plot, y = dataset, colour = feature)) +
  geom_vline(xintercept = 0, linewidth = 0.35, colour = "grey35") +
  geom_errorbar(aes(xmin = low, xmax = high), orientation = "y", width = 0.16, linewidth = 0.45, alpha = 0.78) +
  geom_point(size = 2.7, alpha = 0.95) +
  scale_colour_manual(values = feature_colors, breaks = feature_order) +
  scale_x_continuous(breaks = seq(-1, 1, 0.5)) +
  coord_cartesian(xlim = c(-1, 1)) +
  labs(
    title = "Downstream R2 by dataset",
    subtitle = "Points: mean across properties; bars: mean seed std",
    x = "Mean R2",
    y = "Dataset"
  ) +
  theme_pub(11)

ggsave(
  filename = file.path(out_dir, "r_vibench_dataset_feature_pointrange_std.svg"),
  plot = p_overview,
  width = 8.6,
  height = 5.2
)

# Violin/boxplot across properties. This is a distribution over properties, not seed-level violin.
violin_df <- df
violin_df$dataset <- factor(violin_df$dataset, levels = dataset_order)
violin_df$feature_short <- factor(violin_df$feature_short, levels = feature_short[feature_order])

p_violin <- ggplot(violin_df, aes(x = feature_short, y = r2_plot, fill = feature)) +
  geom_hline(yintercept = 0, linewidth = 0.25, colour = "grey45") +
  geom_violin(width = 0.92, alpha = 0.72, colour = NA, trim = FALSE) +
  geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.95, colour = "grey25", linewidth = 0.25) +
  stat_summary(fun = mean, geom = "point", shape = 21, size = 1.8, fill = "white", colour = "black", stroke = 0.25) +
  facet_wrap(~ dataset, ncol = 4) +
  scale_fill_manual(values = feature_colors, breaks = feature_order) +
  scale_y_continuous(breaks = seq(-1, 1, 0.5)) +
  coord_cartesian(ylim = c(-1, 1)) +
  labs(
    title = "R2 distribution across properties",
    subtitle = "Violin/boxplot uses the 12 property means per dataset",
    x = NULL,
    y = "Mean R2"
  ) +
  theme_pub(10) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1)
  )

ggsave(
  filename = file.path(out_dir, "r_vibench_dataset_feature_violin_properties.svg"),
  plot = p_violin,
  width = 12.2,
  height = 7.2
)

# Dataset-specific property-level point-range plots with seed std.
for (ds in dataset_order) {
  sub <- df[df$dataset == ds, ]
  sub$property <- factor(sub$property, levels = rev(property_order))
  sub$feature <- factor(sub$feature, levels = feature_order)

  p_ds <- ggplot(sub, aes(x = r2_plot, y = property, colour = feature)) +
    geom_vline(xintercept = 0, linewidth = 0.3, colour = "grey35") +
    geom_errorbar(aes(xmin = r2_low, xmax = r2_high), orientation = "y", width = 0.12, linewidth = 0.34, alpha = 0.72) +
    geom_point(size = 2.1, alpha = 0.94) +
    scale_colour_manual(values = feature_colors, breaks = feature_order) +
    scale_x_continuous(breaks = seq(-1, 1, 0.5)) +
    coord_cartesian(xlim = c(-1, 1)) +
    labs(
      title = paste0(ds, ": property-level R2"),
      subtitle = "Points: seed mean; bars: ±1 std",
      x = "Mean R2",
      y = "Property"
    ) +
    theme_pub(10)

  ggsave(
    filename = file.path(out_dir, "by_dataset_pointrange", paste0("r_", ds, "_property_pointrange_std.svg")),
    plot = p_ds,
    width = 8.8,
    height = 6.2
  )
}

heatmap_df <- df
heatmap_df$dataset <- factor(heatmap_df$dataset, levels = dataset_order)
heatmap_df$property <- factor(heatmap_df$property, levels = rev(property_order))
heatmap_df$feature_short <- factor(heatmap_df$feature_short, levels = feature_short[feature_order])

p_heatmap <- ggplot(heatmap_df, aes(x = feature_short, y = property, fill = r2_plot)) +
  geom_tile(colour = "white", linewidth = 0.55) +
  geom_text(aes(label = sprintf("%.2f", r2_mean)), size = 3.0, family = "Arial", colour = "grey12") +
  facet_wrap(~ dataset, ncol = 4) +
  scale_fill_gradient2(
    low = "#B2182B",
    mid = "white",
    high = "#2166AC",
    midpoint = 0,
    limits = c(-1, 1),
    breaks = seq(-1, 1, 0.5),
    name = "Mean R2"
  ) +
  labs(
    title = "Feature comparison by dataset and property",
    x = NULL,
    y = NULL
  ) +
  theme_pub(10) +
  theme(
    legend.position = "right",
    axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1),
    panel.grid = element_blank()
  )

ggsave(
  filename = file.path(out_dir, "heatmaps", "r_feature_property_r2_heatmap.svg"),
  plot = p_heatmap,
  width = 12.6,
  height = 8.4
)

best_idx <- ave(df$r2_mean, df$dataset, df$property, FUN = function(x) x == max(x, na.rm = TRUE))
winner_df <- df[as.logical(best_idx), ]
winner_df <- winner_df[!duplicated(winner_df[c("dataset", "property")]), ]
winner_df$dataset <- factor(winner_df$dataset, levels = dataset_order)
winner_df$property <- factor(winner_df$property, levels = rev(property_order))
winner_df$feature_short <- factor(feature_short[as.character(winner_df$feature)], levels = feature_short[feature_order])

p_winner <- ggplot(winner_df, aes(x = dataset, y = property, fill = feature)) +
  geom_tile(colour = "white", linewidth = 0.65) +
  geom_text(aes(label = feature_short), size = 4.0, family = "Arial", fontface = "bold", colour = "grey10") +
  scale_fill_manual(values = feature_colors, breaks = feature_order) +
  labs(
    title = "Best feature for each dataset-property pair",
    x = "Dataset",
    y = NULL
  ) +
  theme_pub(12) +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 25, hjust = 1, vjust = 1),
    panel.grid = element_blank()
  )

ggsave(
  filename = file.path(out_dir, "heatmaps", "r_winning_feature_tilemap.svg"),
  plot = p_winner,
  width = 9.8,
  height = 6.3
)

wide_morgan <- reshape(
  df[df$feature %in% c("Morgan fingerprint", "Flow + Morgan FP"), c("dataset", "property", "feature", "r2_mean")],
  idvar = c("dataset", "property"),
  timevar = "feature",
  direction = "wide"
)
names(wide_morgan) <- sub("^r2_mean\\.", "", names(wide_morgan))
wide_morgan$delta <- wide_morgan[["Flow + Morgan FP"]] - wide_morgan[["Morgan fingerprint"]]
wide_morgan$dataset <- factor(wide_morgan$dataset, levels = dataset_order)
wide_morgan$property <- factor(wide_morgan$property, levels = rev(property_order))

p_delta <- ggplot(wide_morgan, aes(x = delta, y = property)) +
  geom_vline(xintercept = 0, linewidth = 0.35, colour = "grey35") +
  geom_segment(aes(x = 0, xend = delta, yend = property), linewidth = 0.65, colour = "grey45") +
  geom_point(aes(colour = delta), size = 3.0) +
  facet_wrap(~ dataset, ncol = 4) +
  scale_colour_gradient2(
    low = "#B2182B",
    mid = "grey75",
    high = "#2166AC",
    midpoint = 0,
    name = "Delta R2"
  ) +
  labs(
    title = "Flow+FP gain over Morgan FP",
    x = "Delta mean R2",
    y = NULL
  ) +
  theme_pub(13) +
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold", size = 19, margin = margin(b = 9)),
    strip.text = element_text(face = "bold", size = 13),
    axis.text.x = element_text(size = 11),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 14, face = "bold"),
    legend.text = element_text(size = 11),
    legend.title = element_text(size = 12, face = "bold")
  )

ggsave(
  filename = file.path(out_dir, "dumbbell", "r_flow_fp_gain_over_fp.svg"),
  plot = p_delta,
  width = 13.2,
  height = 8.1
)

radar_radius <- function(x) {
  (pmax(pmin(x, 1), -1) + 1) / 2
}

draw_radar <- function(sub, title, file, width = 10.8, height = 11.8, cex_axis = 2.05, cex_title = 2.65) {
  props <- property_order
  labels <- unname(property_labels[props])
  n <- length(props)
  angles <- seq(pi / 2, pi / 2 - 2 * pi + 2 * pi / n, length.out = n)
  ring_values <- c(-1, -0.5, 0, 0.5, 1)
  ring_radius <- radar_radius(ring_values)

  svg(file, width = width, height = height, family = "Arial")
  par(mar = c(1.0, 1.0, 3.4, 1.0), xpd = NA, family = "Arial")
  plot.new()
  plot.window(xlim = c(-1.60, 1.60), ylim = c(-1.98, 1.38), asp = 1)

  for (rr in ring_radius) {
    polygon(rr * cos(angles), rr * sin(angles), border = "grey82", col = NA, lwd = 0.8)
  }
  for (a in angles) {
    segments(0, 0, cos(a), sin(a), col = "grey86", lwd = 0.8)
  }
  for (i in seq_along(props)) {
    text(1.23 * cos(angles[i]), 1.23 * sin(angles[i]), labels[i], cex = cex_axis, font = 2)
  }
  text(0.06, ring_radius, labels = ring_values, cex = 1.65, col = "grey35", pos = 4)

  sub <- sub[sub$feature %in% feature_order, ]
  for (feat in feature_order) {
    vals <- sub$r2_mean[sub$feature == feat][match(props, as.character(sub$property[sub$feature == feat]))]
    rr <- radar_radius(vals)
    x <- rr * cos(angles)
    y <- rr * sin(angles)
    polygon(x, y, border = adjustcolor(feature_colors[[feat]], alpha.f = 0.95), col = adjustcolor(feature_colors[[feat]], alpha.f = ifelse(feat %in% c("Morgan fingerprint", "Flow + Morgan FP"), 0.035, 0.055)), lwd = 3.2, lty = feature_lty[[feat]])
    if (!is.na(feature_pch[[feat]])) {
      points(x, y, pch = feature_pch[[feat]], bg = "white", col = feature_colors[[feat]], cex = 1.2, lwd = 1.55)
    }
  }

  title(main = title, cex.main = cex_title, font.main = 2, line = 1.1)
  legend(
    x = -1.06,
    y = -1.55,
    legend = feature_order,
    col = feature_colors[feature_order],
    lty = feature_lty[feature_order],
    pch = feature_pch[feature_order],
    pt.bg = "white",
    lwd = 3.0,
    bty = "n",
    horiz = FALSE,
    ncol = 2,
    x.intersp = 0.85,
    y.intersp = 1.35,
    cex = 1.78
  )
  dev.off()
}

for (ds in dataset_order) {
  sub <- df[df$dataset == ds, ]
  draw_radar(
    sub = sub,
    title = ds,
    file = file.path(out_dir, "starplots", paste0("r_", ds, "_starplot.svg"))
  )
}

message("Wrote R plots to: ", normalizePath(out_dir, winslash = "/", mustWork = FALSE))

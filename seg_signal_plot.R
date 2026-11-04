library(dplyr)
library(pheatmap)
library(tidyr)
library(stringr)
library(ggplot2)
library(patchwork)  # Redifine / as putting 2 plots up and down

df_fit_hmm = read.csv("C:/Users/lenovo/Desktop/stageM2/ribostats/ribostats/Segmentation_signal/New_ribohmm/Outputs/df_fit_hmm_fusion.txt",
                      header = TRUE, sep = ",")
df_candidate_orf = read.csv("C:/Users/lenovo/Desktop/stageM2/ribostats/ribostats/Segmentation_signal/New_ribohmm/Outputs/candidate_orfs_fusion.txt",
                            header = TRUE, sep = ",")
cm = read.csv("C:/Users/lenovo/Desktop/stageM2/ribostats/ribostats/data_i2bc/kmer-counts-fusions.tsv/kmer-counts-fusions.tsv",header = TRUE, sep = "\t")






#GENE_NAME = "IMPDH1--IMPDH2"

GENE_NAME = "EEF1D--AC010616.2"
signal_df <- cm[grepl(GENE_NAME, cm$id), ]
signal_df$k_idx <- as.integer(sub(".*_kmer_", "", signal_df$id))

xmin <- min(df_fit_hmm$start_kidx, na.rm = TRUE)
xmax <- max(df_fit_hmm$end_kidx, na.rm = TRUE)


# Original signal
p1 = ggplot(signal_df, aes(x = k_idx, y=sum))+
  geom_line()+
  theme_bw()+
  labs(
    title = paste0(GENE_NAME," Signal"),
    x = NULL,
    y= "sum"
  )+
  scale_x_continuous(
    limits = c(xmin, xmax),
    expand = c(0, 0)
  )
p1

# log signal
p2 = ggplot(signal_df,aes(x=k_idx, y = log(sum)))+
  geom_line()+
  theme_bw()+
  labs(
    title = NULL,
    x = NULL,
    y= "log sum"
  )+
  scale_x_continuous(
    limits = c(xmin, xmax),
    expand = c(0, 0)
  ) 
p2 

df_fit_hmm = df_fit_hmm[grepl(GENE_NAME, df_fit_hmm$gene_name), ]
p3 = ggplot(df_fit_hmm, aes(x = start_kidx, y = 1, color = state_hmm)) +
  geom_point(size = 1) + theme_bw()+ 
  labs( title = "Corrected best state at each position", 
        x = "Position", y = NULL ) + 
  theme( axis.text.y = element_blank(), axis.ticks.y = element_blank() )+
  scale_x_continuous(limits = c(xmin, xmax),expand = c(0, 0)) 
p3

df_candidate_orf = df_candidate_orf[grepl(GENE_NAME,df_candidate_orf$Gene),]
df_candidate_orf$frame <- as.integer(df_candidate_orf$frame)


df_candidate_orf$ymin <- df_candidate_orf$frame
df_candidate_orf$ymax <- df_candidate_orf$frame + 0.8
df_candidate_orf$ytext <- df_candidate_orf$ymax + 0.08

p4<- ggplot(df_candidate_orf) +
  geom_rect(aes(xmin = k_start,
                xmax = k_end,
                ymin = ymin,
                ymax = ymax),
            fill = "steelblue",
            color = "gray40",
            alpha = 0.8) +
  geom_text(aes(x = k_start, y = ytext, label = k_start),
            size = 3, hjust = 0) +
  geom_text(aes(x = k_end, y = ytext, label = k_end),
            size = 3, hjust = 1) +
  scale_x_continuous(limits = c(xmin, xmax), expand = c(0, 0)) +
  scale_y_continuous(breaks = c(0.4, 1.4, 2.4),
                     labels = c("frame 0", "frame 1", "frame 2")) +
  labs(title = "Predicted ORFs",
       x = "Position",
       y = NULL) +
  theme_bw()

p4
p_final <- p1/p2/p3/p4 + patchwork::plot_layout(heights = c(2,2,1,5))
p_final










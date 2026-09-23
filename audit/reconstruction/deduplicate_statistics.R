suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(lmerTest))
suppressPackageStartupMessages(library(emmeans))
out <- 'D:/SummerResearch/audit/reconstruction/'
raw <- read.csv('D:/Rprojects/research_data_synthesis/April29_macro_max_mfa.csv')
analyse <- function(input, title, random_method=FALSE) {
  cat('\nCASE:',title,'\n')
  for(n in c('dataset','method','has_y','tvae','approach','y_synth')) input[[n]] <- factor(input[[n]])
  input$appr_has_y <- factor(ifelse(input$approach=='old','old',ifelse(input$has_y=='0','new_no_y','new_has_y')),levels=c('old','new_no_y','new_has_y'))
  print(table(input$dataset));print(aggregate(macro_max~approach,data=input,FUN=mean))
  ds <- reshape(aggregate(macro_max~dataset+approach,data=input,FUN=mean),idvar='dataset',timevar='approach',direction='wide')
  ds$delta <- ds$macro_max.new-ds$macro_max.old
  print(ds)
  cat('MEAN DIFFERENCE',mean(ds$delta),'RELATIVE',mean(ds$delta)/mean(ds$macro_max.old),'\n')
  f <- lmer(macro_max~appr_has_y+(1|dataset)+(1|tvae),data=input,REML=FALSE)
  cat('ORIGINAL MODEL, SAME THREE CONTRASTS / SIDAK ADJUSTMENT\n')
  contrasts <- contrast(emmeans(f,~appr_has_y),method=list(new_hasY_vs_new_noY=c(0,-1,1),newApproach_vs_oldApproach=c(-1,.5,.5),hasY_vs_no_Y=c(1,-2,1)),adjust='sidak')
  print(summary(contrasts,infer=c(TRUE,TRUE)));print(VarCorr(f));cat('SINGULAR',isSingular(f),'\n')
  cat('PAIRED DATASET-LEVEL TEST; TWO-SIDED\n')
  print(t.test(ds$delta,mu=0))
  if(random_method) {
    cat('INCLUDING LABELING-METHOD RANDOM EFFECT\n')
    fy <- lmer(macro_max~appr_has_y+(1|dataset)+(1|tvae)+(1|y_synth),data=input,REML=FALSE)
    print(summary(contrast(emmeans(fy,~appr_has_y),list(new_vs_old=c(-1,.5,.5))),infer=c(TRUE,TRUE)))
    print(VarCorr(fy));cat('SINGULAR',isSingular(fy),'\n')
  }
  return(invisible(ds))
}
nog <- raw[!grepl('gauss',raw$method),]
collapsed <- raw
collapsed$dataset[collapsed$dataset %in% c('Census','Census_Kdd')] <- 'Adult'
collapsed <- aggregate(macro_max~method+has_y+tvae+approach+dataset+y_synth,data=collapsed,FUN=mean)
write.csv(collapsed,paste0(out,'deduplicated_method_scores.csv'),row.names=FALSE)
analyse(nog,'Original six labels; Gaussian excluded')
d <- analyse(collapsed[!grepl('gauss',collapsed$method),],'Four datasets; average Adult aliases; Gaussian excluded',TRUE)
write.csv(d,paste0(out,'deduplicated_dataset_differences.csv'),row.names=FALSE)
analyse(nog[!nog$dataset %in% c('Census','Census_Kdd'),],'Four datasets; retain only Adult run; Gaussian excluded')
analyse(collapsed,'Four datasets; average Adult aliases; Gaussian INCLUDED')

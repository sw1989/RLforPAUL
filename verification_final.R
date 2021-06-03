### This is the R script for user simulator verification reported in Wang, Zhang, Krose, & van Hoof, 2021
## Function for computing probability of running given the three key parameters
computeRunProb <- function(memory, urge, context, default_prob) {
	prob_run <- memory * urge * context * default_prob
	if (prob_run > 1) {
		prob_run <- 1
		print("There is an error in computing run probability")
	}
	return (prob_run)
}

## Function for simulating the running behavior of a single user
simulateOnePerson <- function(memory_scale, urge_scale, context, notif_type, maximum_DP, night_break, default_prob) {
	memory <- rep(NA, maximum_DP*7)
	memory[1] <- memory_scale ** sample(c(0:20), 1)
	urge <- rep(NA, maximum_DP*7)
	urge[1] <- 1
	context <- rep(context, maximum_DP*7)
	notif <- rep(0, maximum_DP*7)
	notif_day_pattern <- rep(0, maximum_DP)
	if (notif_type == "random_week") {
		notif[sample(c(1:(maximum_DP*7)), 14, replace=FALSE)] <- 1
	} else if (notif_type == "random_day") {
		notif_day_pattern[sample(c(1:12), 2)] <- 1
		notif <- rep(notif_day_pattern, 7)
	} else {
		notif_day_pattern[c(5, 9)] <- 1
		notif <- rep(notif_day_pattern, 7)
	}
	run_prob <- rep(NA, maximum_DP*7)
	run <- rep(NA, maximum_DP*7)
	for (i in 1:(maximum_DP*7)) {
		if (i != 1) {
			if (notif[i] == 1) {
				memory[i] <- 1
			} else {
				if (night_break && (i %% maximum_DP == 1)) {
					memory[i] <- memory[i-1] * (memory_scale ** 12)
				} else {
					memory[i] <- memory[i-1] * memory_scale
				}	
			}
			if (night_break && (i %% maximum_DP == 1)) {
				urge[i] <- urge[i-1] + urge_scale * 12
				if (urge[i] > 1) {
					urge[i] <- 1
				}
			} else {
				if (run[i-1] == 1) {
					urge[i] <- 0.001
				} else {
					urge[i] <- urge[i-1] + urge_scale
					if (urge[i] > 1) {
						urge[i] <- 1
					}
				}
			}
		}
		run_prob[i] <- computeRunProb(memory[i], urge[i], context[i], default_prob=default_prob)
		run[i] <- sample(c(0, 1), 1, prob=c((1-run_prob[i]), run_prob[i]))
	}
	return (list(memory=memory, urge=urge, context=context, notif=notif, run_prob=run_prob, run=run))
}

## Function for generating simulation data 
simulateData <- function(memory_scale, urge_scale, context, notif_type, maximum_DP, night_break, default_prob, n_person) {
	data <- data.frame(ppn=c(1:(maximum_DP*7*n_person)), day=c(rep(1, maximum_DP), rep(2, maximum_DP), rep(3, maximum_DP), rep(4, maximum_DP), rep(5, maximum_DP), rep(6, maximum_DP), rep(7, maximum_DP)),
		DP=rep(c(1:maximum_DP), 7*n_person), memory=rep(NA, maximum_DP*7*n_person), urge=rep(NA, maximum_DP*7*n_person), context=rep(NA, maximum_DP*7*n_person), notif=rep(NA, maximum_DP*7*n_person),
		run_prob=rep(NA, maximum_DP*7*n_person), run=rep(NA, maximum_DP*7*n_person), index=rep(c(1:(maximum_DP*7)), n_person))
	plot <- list(list())
	data$ppn <- (data$ppn - 1) %/% (maximum_DP*7) + 1
	for (ppn in unique(data$ppn)) {
		result <- simulateOnePerson(memory_scale=memory_scale, urge_scale=urge_scale, context=context, notif_type=notif_type, maximum_DP=maximum_DP, night_break=night_break, default_prob=default_prob)
		for (var in names(data)[4:9]) {
			data[[var]][which(data$ppn==ppn)] <- result[[var]]
		}
		## Plot individual data
		plot_data <- data[which(data$ppn==ppn),]
		for (var in c("memory", "urge", "run", "run_prob")) {
			plot[[as.character(ppn)]][[var]] <- ggplot(plot_data, aes_string(x="index", y=var)) + geom_point(size=1.5) + geom_vline(xintercept=c(16.5, 32.5, 48.5, 64.5, 80.5, 96.5), linetype="dashed")
			if (var != "run") {
				plot[[as.character(ppn)]][[var]] <- plot[[as.character(ppn)]][[var]] + geom_line()
			}
			if (var != "urge") {
				plot[[as.character(ppn)]][[var]] <- plot[[as.character(ppn)]][[var]] + geom_vline(xintercept=plot_data$index[which(plot_data$notif==1)], color="red")
			}
		}
	}
	return(list(data=data, plot=plot))
}

## Start simulating the results: relationship between parameter values and some running behavior outcomes
set.seed(2019)
library("ggplot2")
theme_set(theme_bw())
# Change memory decay parameter
default_prob <- 0.1
maximum_DP <- 12
n_person <- 100
night_break <- TRUE
urge_scale <- 0.05
context <- 1
df <- data.frame(memory_scale=rep(seq(0, 1, 0.1), 3), notif_type=c(rep("random_week", 11), rep("random_day", 11), rep("fixed", 11)))
var_outcome <- c("run_frequency", "percent_on_notif", "prob_adjacent", "prob_adjacent_night")
for (var in var_outcome) {
	df[[var]] <- rep(NA, 11*3)
}
for (i in 1:nrow(df)) {
	print(i)
	result <- simulateData(memory_scale=df$memory_scale[i], urge_scale=urge_scale, context=context, notif_type=df$notif_type[i], maximum_DP=maximum_DP, night_break=night_break, default_prob=default_prob, n_person=n_person)$data
	df$run_frequency[i] <- sum(result$run) / nrow(result) * maximum_DP
	on_notif <- length(intersect(which(result$run==1), which(result$notif==1)))
	count <- 0
	run_night <- 0
	count_night <- 0
	for (j in 1:nrow(result)) {
		if (result$index[j] < (maximum_DP*7-1)) {
			if (result$run[j] == 1 && result$run[j+2] == 1) {
				count <- count + 1
			}
			if (result$run[j] == 1 && result$DP[j] == maximum_DP) {
				run_night <- run_night + 1
				if (result$run[j+2] == 1) {
					count_night <- count_night + 1
				}
			}
		}
	}
	if (sum(result$run) != 0) {
		df$prob_adjacent[i] <- count / sum(result$run)
		df$percent_on_notif[i] <- on_notif / sum(result$run)
	}
	if (run_night != 0) {
		df$prob_adjacent_night[i] <- count_night / run_night
	}
}

names(df)[2] <- "Notification"
df[["Notification"]] <- as.character(df[["Notification"]])
df[["Notification"]][which(df[["Notification"]]=="fixed")] <- "Fixed"
df[["Notification"]][which(df[["Notification"]]=="random_day")] <- "Random Day"
df[["Notification"]][which(df[["Notification "]]=="random_week")] <- "Random Week"
df[["Notification "]] <- factor(df[["Notification"]], levels=c("Fixed", "Random Day", "Random Week"))

p_result <- list()
for (var in var_outcome) {
	p_result[[var]] <- ggplot(df, aes_string(x="memory_scale", y=var, group="Notification", color="Notification ")) + geom_point(size=2) + geom_line()
	if (var != "run_frequency") {
		p_result[[var]] <- p_result[[var]] + ylim(0, 1)
	}
}

p_result[["run_frequency"]] <- p_result[["run_frequency"]] + ylab("Run per Day") + geom_line(size=1) + geom_point(size=2) + theme(legend.text=element_text(size=16)) + xlab("Memory Retention Rate") +
      theme(legend.position="bottom", legend.title = element_text(size=13), legend.text = element_text(size=13)) +
      theme(axis.text.x = element_text(size = 13)) + 
      theme(axis.text.y = element_text(size = 13)) +
      theme(axis.title.y = element_text(size = 14)) +
      theme(axis.title.x = element_text(size = 14)) +
      theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

p_result[["percent_on_notif"]] <- p_result[["percent_on_notif"]] + ylab("Percent on Notif") + geom_line(size=1) + geom_point(size=2) + theme(legend.text=element_text(size=16)) + xlab("Memory Retention Rate") +
      theme(legend.position="bottom", legend.title = element_text(size=13), legend.text = element_text(size=13)) +
      theme(axis.text.x = element_text(size = 13)) + 
      theme(axis.text.y = element_text(size = 13)) +
      theme(axis.title.y = element_text(size = 14)) +
      theme(axis.title.x = element_text(size = 14)) +
      theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

setwd("D:\\Work TUe\\PhD\\Projects\\Collaborations\\Modeling PA with Sihan\\for chao")
tiff(filename = "Figure 1A.tiff", width = 1500, height = 900, units = "px", res = 300)
print(p_result[["run_frequency"]])
graphics.off()

tiff(filename = "Figure 2A.tiff", width = 1500, height = 900, units = "px", res = 300)
print(p_result[["percent_on_notif"]])
graphics.off()


# Change urge recovery parameter
night_break <- TRUE
memory_scale <- 0.8
df <- data.frame(urge_scale=rep(seq(0, 0.5, 0.05), 3), notif_type=c(rep("random_week", 11), rep("random_day", 11), rep("fixed", 11)))
var_outcome <- c("run_frequency", "percent_on_notif", "prob_adjacent", "prob_adjacent_night")
for (var in var_outcome) {
	df[[var]] <- rep(NA, 11*3)
}
for (i in 1:nrow(df)) {
	print(i)
	result <- simulateData(memory_scale=memory_scale, urge_scale=df$urge_scale[i], context=context, notif_type=df$notif_type[i], maximum_DP=maximum_DP, night_break=night_break, default_prob=default_prob, n_person=n_person)$data
	df$run_frequency[i] <- sum(result$run) / nrow(result) * maximum_DP
	on_notif <- length(intersect(which(result$run==1), which(result$notif==1)))
	count <- 0
	run_night <- 0
	count_night <- 0
	for (j in 1:nrow(result)) {
		if (result$index[j] < (maximum_DP*7-1)) {
			if (result$run[j] == 1 && result$run[j+2] == 1) {
				count <- count + 1
			}
			if (result$run[j] == 1 && result$DP[j] == maximum_DP) {
				run_night <- run_night + 1
				if (result$run[j+2] == 1) {
					count_night <- count_night + 1
				}
			}
		}
	}
	if (sum(result$run) != 0) {
		df$prob_adjacent[i] <- count / sum(result$run)
		df$percent_on_notif[i] <- on_notif / sum(result$run)
	}
	if (run_night != 0) {
		df$prob_adjacent_night[i] <- count_night / run_night
	}
}

names(df)[2] <- "Notification"
df[["Notification"]] <- as.character(df[["Notification"]])
df[["Notification"]][which(df[["Notification"]]=="fixed")] <- "Fixed"
df[["Notification"]][which(df[["Notification"]]=="random_day")] <- "Random Day"
df[["Notification"]][which(df[["Notification "]]=="random_week")] <- "Random Week"
df[["Notification "]] <- factor(df[["Notification"]], levels=c("Fixed", "Random Day", "Random Week"))

p_result <- list()
for (var in var_outcome) {
	p_result[[var]] <- ggplot(df, aes_string(x="urge_scale", y=var, group="Notification", color="Notification ")) + geom_point(size=2) + geom_line()
	if (var != "run_frequency") {
		p_result[[var]] <- p_result[[var]] + ylim(0, 1)
	}
}

p_result[["run_frequency"]] <- p_result[["run_frequency"]] + ylab("Run per Day") + geom_line(size=1) + geom_point(size=2) + theme(legend.text=element_text(size=16)) + xlab("Urge Recovery Rate") +
      theme(legend.position="bottom", legend.title = element_text(size=13), legend.text = element_text(size=13)) +
      theme(axis.text.x = element_text(size = 13)) + 
      theme(axis.text.y = element_text(size = 13)) +
      theme(axis.title.y = element_text(size = 14)) +
      theme(axis.title.x = element_text(size = 14)) +
      theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

p_result[["percent_on_notif"]] <- p_result[["percent_on_notif"]] + ylab("Percent on Notif") + geom_line(size=1) + geom_point(size=2) + theme(legend.text=element_text(size=16)) + xlab("Urge Recovery Rate") +
      theme(legend.position="bottom", legend.title = element_text(size=13), legend.text = element_text(size=13)) +
      theme(axis.text.x = element_text(size = 13)) + 
      theme(axis.text.y = element_text(size = 13)) +
      theme(axis.title.y = element_text(size = 14)) +
      theme(axis.title.x = element_text(size = 14)) +
      theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

setwd("D:\\Work TUe\\PhD\\Projects\\Collaborations\\Modeling PA with Sihan\\for chao")
tiff(filename = "Figure 1B.tiff", width = 1500, height = 900, units = "px", res = 300)
print(p_result[["run_frequency"]])
graphics.off()

tiff(filename = "Figure 2B.tiff", width = 1500, height = 900, units = "px", res = 300)
print(p_result[["percent_on_notif"]])
graphics.off()

# Change context desirability
memory_scale <- 0.8
urge_scale <- 0.05
night_break <- TRUE
df <- data.frame(context=rep(seq(0.2, 5, 0.2), 3), notif_type=c(rep("random_week", 25), rep("random_day", 25), rep("fixed", 25)))
var_outcome <- c("run_frequency", "percent_on_notif", "prob_adjacent", "prob_adjacent_night")
for (var in var_outcome) {
	df[[var]] <- rep(NA, 25*3)
}
for (i in 1:nrow(df)) {
	print(i)
	result <- simulateData(memory_scale=memory_scale, urge_scale=urge_scale, context=df$context[i], notif_type=df$notif_type[i], maximum_DP=maximum_DP, night_break=night_break, default_prob=default_prob, n_person=n_person)$data
	df$run_frequency[i] <- sum(result$run) / nrow(result) * maximum_DP
	on_notif <- length(intersect(which(result$run==1), which(result$notif==1)))
	count <- 0
	run_night <- 0
	count_night <- 0
	for (j in 1:nrow(result)) {
		if (result$index[j] < (maximum_DP*7-1)) {
			if (result$run[j] == 1 && result$run[j+2] == 1) {
				count <- count + 1
			}
			if (result$run[j] == 1 && result$DP[j] == maximum_DP) {
				run_night <- run_night + 1
				if (result$run[j+2] == 1) {
					count_night <- count_night + 1
				}
			}
		}
	}
	if (sum(result$run) != 0) {
		df$prob_adjacent[i] <- count / sum(result$run)
		df$percent_on_notif[i] <- on_notif / sum(result$run)
	}
	if (run_night != 0) {
		df$prob_adjacent_night[i] <- count_night / run_night
	}
}

names(df)[2] <- "Notification"
df[["Notification"]] <- as.character(df[["Notification"]])
df[["Notification"]][which(df[["Notification"]]=="fixed")] <- "Fixed"
df[["Notification"]][which(df[["Notification"]]=="random_day")] <- "Random Day"
df[["Notification"]][which(df[["Notification"]]=="random_week")] <- "Random Week"
df[["Notification "]] <- factor(df[["Notification"]], levels=c("Fixed", "Random Day", "Random Week"))

p_result <- list()
for (var in var_outcome) {
	p_result[[var]] <- ggplot(df, aes_string(x="context", y=var, group="Notification", color="Notification ")) + geom_point(size=2) + geom_line()
	if (var != "run_frequency") {
		p_result[[var]] <- p_result[[var]] + ylim(0, 1)
	}
}

p_result[["run_frequency"]] <- p_result[["run_frequency"]] + ylab("Run per Day") + geom_line(size=1) + geom_point(size=2) + theme(legend.text=element_text(size=16)) + xlab("Context Desirability") +
      theme(legend.position="bottom", legend.title = element_text(size=13), legend.text = element_text(size=13)) +
      theme(axis.text.x = element_text(size = 13)) + 
      theme(axis.text.y = element_text(size = 13)) +
      theme(axis.title.y = element_text(size = 14)) +
      theme(axis.title.x = element_text(size = 14)) +
      theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

p_result[["percent_on_notif"]] <- p_result[["percent_on_notif"]] + ylab("Percent on Notif") + geom_line(size=1) + geom_point(size=2) + theme(legend.text=element_text(size=16)) + xlab("Context Desirability") +
      theme(legend.position="bottom", legend.title = element_text(size=13), legend.text = element_text(size=13)) +
      theme(axis.text.x = element_text(size = 13)) + 
      theme(axis.text.y = element_text(size = 13)) +
      theme(axis.title.y = element_text(size = 14)) +
      theme(axis.title.x = element_text(size = 14)) +
      theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

setwd("D:\\Work TUe\\PhD\\Projects\\Collaborations\\Modeling PA with Sihan\\for chao")
tiff(filename = "Figure 1C.tiff", width = 1500, height = 900, units = "px", res = 300)
print(p_result[["run_frequency"]])
graphics.off()

tiff(filename = "Figure 2C.tiff", width = 1500, height = 900, units = "px", res = 300)
print(p_result[["percent_on_notif"]])
graphics.off()
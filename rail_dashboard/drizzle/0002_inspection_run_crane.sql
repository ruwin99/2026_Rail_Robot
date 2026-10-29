ALTER TABLE `inspection_runs` ADD `crane_id` text;
--> statement-breakpoint
ALTER TABLE `inspection_runs` ADD `crane_label` text;
--> statement-breakpoint
CREATE INDEX `inspection_runs_crane_date_idx` ON `inspection_runs` (`crane_id`,`local_date`);

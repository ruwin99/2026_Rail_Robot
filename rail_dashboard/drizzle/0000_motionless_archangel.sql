CREATE TABLE `analyses` (
	`capture_id` text NOT NULL,
	`defect_type` text NOT NULL,
	`status` text NOT NULL,
	`detected` integer,
	`positive_pixels` integer,
	`inspected_pixels` integer,
	`ratio_fraction` real,
	`detector_method` text,
	`provenance_id` text,
	`overlap_policy` text,
	`grade_pixel_counts_json` text,
	PRIMARY KEY(`capture_id`, `defect_type`),
	FOREIGN KEY (`capture_id`) REFERENCES `captures`(`capture_id`) ON UPDATE no action ON DELETE cascade,
	FOREIGN KEY (`provenance_id`) REFERENCES `model_provenance`(`provenance_id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `artifacts` (
	`artifact_id` text PRIMARY KEY NOT NULL,
	`capture_id` text NOT NULL,
	`artifact_type` text NOT NULL,
	`object_key` text NOT NULL,
	`public_url` text,
	`media_type` text NOT NULL,
	`width` integer NOT NULL,
	`height` integer NOT NULL,
	`sha256` text NOT NULL,
	FOREIGN KEY (`capture_id`) REFERENCES `captures`(`capture_id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `artifacts_capture_type_uq` ON `artifacts` (`capture_id`,`artifact_type`);--> statement-breakpoint
CREATE TABLE `captures` (
	`capture_id` text PRIMARY KEY NOT NULL,
	`run_id` text NOT NULL,
	`phase` text NOT NULL,
	`phase_sequence` integer,
	`logical_zone_number` integer,
	`trigger` text NOT NULL,
	`captured_at_utc` text NOT NULL,
	`captured_local_date` text NOT NULL,
	`display_timezone` text NOT NULL,
	`width` integer NOT NULL,
	`height` integer NOT NULL,
	`processing_status` text NOT NULL,
	`raw_image_key` text NOT NULL,
	FOREIGN KEY (`run_id`) REFERENCES `inspection_runs`(`run_id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `captures_run_phase_idx` ON `captures` (`run_id`,`phase`);--> statement-breakpoint
CREATE INDEX `captures_local_date_idx` ON `captures` (`captured_local_date`);--> statement-breakpoint
CREATE UNIQUE INDEX `captures_run_phase_sequence_uq` ON `captures` (`run_id`,`phase`,`phase_sequence`);--> statement-breakpoint
CREATE TABLE `inspection_runs` (
	`run_id` text PRIMARY KEY NOT NULL,
	`schema_version` integer NOT NULL,
	`pipeline_version` integer NOT NULL,
	`started_at_utc` text,
	`finished_at_utc` text,
	`display_timezone` text NOT NULL,
	`local_date` text,
	`status` text NOT NULL,
	`capture_target` integer NOT NULL,
	`failure_reason` text
);
--> statement-breakpoint
CREATE INDEX `inspection_runs_local_date_idx` ON `inspection_runs` (`local_date`);--> statement-breakpoint
CREATE TABLE `model_provenance` (
	`provenance_id` text PRIMARY KEY NOT NULL,
	`run_id` text NOT NULL,
	`role` text NOT NULL,
	`model_filename` text,
	`model_sha256` text,
	`detector_method` text,
	`probability_threshold` real,
	`min_component_pixels` integer,
	`preprocessing` text,
	`input_contract` text,
	`output_contract` text,
	FOREIGN KEY (`run_id`) REFERENCES `inspection_runs`(`run_id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `model_provenance_run_role_uq` ON `model_provenance` (`run_id`,`role`);--> statement-breakpoint
CREATE TABLE `run_summaries` (
	`run_id` text PRIMARY KEY NOT NULL,
	`initial_capture_count` integer NOT NULL,
	`rescan_capture_count` integer NOT NULL,
	`before_positive_pixels` integer NOT NULL,
	`before_inspected_pixels` integer NOT NULL,
	`before_ratio_fraction` real,
	`after_positive_pixels` integer NOT NULL,
	`after_inspected_pixels` integer NOT NULL,
	`after_ratio_fraction` real,
	`absolute_reduction_fraction` real,
	`relative_improvement_fraction` real,
	`summary_complete` integer NOT NULL,
	FOREIGN KEY (`run_id`) REFERENCES `inspection_runs`(`run_id`) ON UPDATE no action ON DELETE cascade
);

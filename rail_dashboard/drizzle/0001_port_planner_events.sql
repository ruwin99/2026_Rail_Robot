CREATE TABLE `planner_events` (
	`event_id` text PRIMARY KEY NOT NULL,
	`event_date` text NOT NULL,
	`start_time` text NOT NULL,
	`end_time` text NOT NULL,
	`title` text NOT NULL,
	`category` text NOT NULL,
	`location` text,
	`notes` text,
	`status` text NOT NULL,
	`source` text NOT NULL,
	`created_at_utc` text NOT NULL,
	`updated_at_utc` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `planner_events_date_idx` ON `planner_events` (`event_date`);

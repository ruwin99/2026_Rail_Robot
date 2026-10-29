DROP INDEX `captures_run_phase_sequence_uq`;
--> statement-breakpoint
ALTER TABLE `captures` ADD `camera_role` text DEFAULT 'side' NOT NULL;
--> statement-breakpoint
CREATE UNIQUE INDEX `captures_run_phase_sequence_camera_role_uq` ON `captures` (`run_id`,`phase`,`phase_sequence`,`camera_role`);

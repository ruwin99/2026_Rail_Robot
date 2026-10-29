# Completion time; source_stamp identifies the captured original.
std_msgs/Header header
builtin_interfaces/Time source_stamp
string camera_role
string stream_id
uint64 sequence
string model
# Unique per detector.detect() invocation across vision process restarts.
# Shared multitask outputs carry the same ID; count distinct IDs for FPS.
string execution_id
uint64 processing_start_monotonic_ns
uint64 processing_end_monotonic_ns
# Duration of the synchronous detector.detect() call, including its internal work.
float64 processing_ms

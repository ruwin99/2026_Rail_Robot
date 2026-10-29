# One newly evaluated detector result; never a cached result with a new stamp.
std_msgs/Header header
string camera_role
string stream_id
uint64 sequence
string detector
bool valid
string error
string detector_method
int8 rust_grade
float64 crack_ratio
bool obstacle_detected
sensor_msgs/Image mask
float32[] boxes_xyxy
float32[] confidences
int32[] class_ids

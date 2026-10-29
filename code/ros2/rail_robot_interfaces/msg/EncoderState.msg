# Header is Jetson reception time; mcu_millis is the actual STM sample tick.
std_msgs/Header header
string session_id
uint32 sequence
uint32 mcu_millis
# Raw hardware counters wrap at the advertised modulus; these are not distances.
uint32 left_raw_count
uint32 right_raw_count
uint32 left_counter_modulus
uint32 right_counter_modulus
string commanded_direction
float64 commanded_speed_percent
string applied_direction
float64 applied_speed_percent
uint32 left_pwm_compare
uint32 right_pwm_compare
# bit 0 front DC, 1 side DC, 2 front pump, 3 side pump.
uint8 active_output_mask
bool feedback_valid
string error

"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(_runtime_version.Domain.PUBLIC, 5, 29, 0, '', 'can.proto')
_sym_db = _symbol_database.Default()
from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\tcan.proto\x12\x08intercom\x1a\x1fgoogle/protobuf/timestamp.proto"\xa7\x05\n\rVehicleMotion\x12.\n\ntime_stamp\x18\x01 \x01(\x0b2\x1a.google.protobuf.Timestamp\x12\x19\n\x11session_timestamp\x18\x13 \x01(\x04\x12\x16\n\x0eacceleration_x\x18\x02 \x01(\x01\x12\x16\n\x0eacceleration_y\x18\x03 \x01(\x01\x12\x16\n\x0eacceleration_z\x18\x04 \x01(\x01\x12\r\n\x05speed\x18\x05 \x01(\x02\x12\x16\n\x0ewheel_speed_fr\x18\x06 \x01(\x02\x12\x16\n\x0ewheel_speed_fl\x18\x07 \x01(\x02\x12\x16\n\x0ewheel_speed_rr\x18\x08 \x01(\x02\x12\x16\n\x0ewheel_speed_rl\x18\t \x01(\x02\x12\x16\n\x0esteering_angle\x18\n \x01(\x02\x12!\n\x19steering_angle_normalized\x18\x0b \x01(\x02\x12\x1c\n\x14gas_pedal_normalized\x18\x0c \x01(\x02\x12\x1e\n\x16brake_pedal_normalized\x18\r \x01(\x02\x12\x1e\n\x16instructor_pedals_used\x18\x12 \x01(\x08\x12*\n\x04gear\x18\x0f \x01(\x0e2\x1c.intercom.VehicleMotion.Gear\x12\x17\n\x0fspeed_dashboard\x18\x10 \x01(\x02\x12\x0f\n\x07mileage\x18\x11 \x01(\x01\x12\x1d\n\x15speed_limiter_enabled\x18\x14 \x01(\x08\x12\x1c\n\x14speed_limiter_target\x18\x15 \x01(\r\x12\x1e\n\x16cruise_control_enabled\x18\x16 \x01(\x08\x12\x1d\n\x15cruise_control_target\x18\x17 \x01(\r")\n\x04Gear\x12\x05\n\x01P\x10\x00\x12\x05\n\x01R\x10\x01\x12\x05\n\x01N\x10\x02\x12\x05\n\x01D\x10\x03\x12\x05\n\x01B\x10\x04"\xa7\x04\n\x0cVehicleState\x12.\n\ntime_stamp\x18\x01 \x01(\x0b2\x1a.google.protobuf.Timestamp\x12\x19\n\x11session_timestamp\x18\x13 \x01(\x04\x12\x16\n\x0edoor_closed_fl\x18\x02 \x01(\x08\x12\x16\n\x0edoor_closed_fr\x18\x03 \x01(\x08\x12\x16\n\x0edoor_closed_rl\x18\x04 \x01(\x08\x12\x16\n\x0edoor_closed_rr\x18\x05 \x01(\x08\x12\x1a\n\x12driver_seatbelt_on\x18\x06 \x01(\x08\x12\x1d\n\x15passenger_seatbelt_on\x18\x07 \x01(\x08\x12\x1e\n\x16headlight_high_beam_on\x18\x08 \x01(\x08\x126\n\x0bturn_signal\x18\t \x01(\x0e2!.intercom.VehicleState.TurnSignal\x125\n\x0bwipers_mode\x18\n \x01(\x0e2 .intercom.VehicleState.WiperMode\x12&\n\x1ewipers_rain_detect_sensitivity\x18\x0b \x01(\r\x12\x15\n\rwipers_parked\x18\x0c \x01(\x08"*\n\nTurnSignal\x12\x07\n\x03OFF\x10\x00\x12\x08\n\x04LEFT\x10\x01\x12\t\n\x05RIGHT\x10\x02"7\n\tWiperMode\x12\x0c\n\x08DISABLED\x10\x00\x12\x08\n\x04AUTO\x10\x01\x12\x08\n\x04SLOW\x10\x02\x12\x08\n\x04FAST\x10\x03B\x06Z\x04.;pbb\x06proto3')
_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'can_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    _globals['DESCRIPTOR']._loaded_options = None
    _globals['DESCRIPTOR']._serialized_options = b'Z\x04.;pb'
    _globals['_VEHICLEMOTION']._serialized_start = 57
    _globals['_VEHICLEMOTION']._serialized_end = 736
    _globals['_VEHICLEMOTION_GEAR']._serialized_start = 695
    _globals['_VEHICLEMOTION_GEAR']._serialized_end = 736
    _globals['_VEHICLESTATE']._serialized_start = 739
    _globals['_VEHICLESTATE']._serialized_end = 1290
    _globals['_VEHICLESTATE_TURNSIGNAL']._serialized_start = 1191
    _globals['_VEHICLESTATE_TURNSIGNAL']._serialized_end = 1233
    _globals['_VEHICLESTATE_WIPERMODE']._serialized_start = 1235
    _globals['_VEHICLESTATE_WIPERMODE']._serialized_end = 1290
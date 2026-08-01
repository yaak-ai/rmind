"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(_runtime_version.Domain.PUBLIC, 5, 29, 0, '', 'sensor.proto')
_sym_db = _symbol_database.Default()
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0csensor.proto\x12\x08intercom\x1a\x1bgoogle/protobuf/empty.proto\x1a\x1fgoogle/protobuf/timestamp.proto"s\n\x15StartRecordingRequest\x12\x19\n\x0cbitrate_kbps\x18\x01 \x01(\x05H\x00\x88\x01\x01\x12\x1b\n\x0emaxlength_mins\x18\x02 \x01(\x05H\x01\x88\x01\x01B\x0f\n\r_bitrate_kbpsB\x11\n\x0f_maxlength_mins"F\n\x17StartRecordingRequestV2\x12\x13\n\x0bbitrate_bps\x18\x01 \x01(\x05\x12\x16\n\x0emaxlength_mins\x18\x02 \x01(\x05"i\n\rSessionStatus\x12.\n\ntime_stamp\x18\x01 \x01(\x0b2\x1a.google.protobuf.Timestamp\x12\x14\n\x0cis_recording\x18\x02 \x01(\x08\x12\x12\n\ndrive_name\x18\x03 \x01(\t"_\n\x08Incident\x12\x12\n\ndrive_name\x18\x01 \x01(\t\x12\x13\n\x0bincident_id\x18\x02 \x01(\t\x12\x15\n\rstart_time_ns\x18\x03 \x01(\x04\x12\x13\n\x0bend_time_ns\x18\x04 \x01(\x04"0\n\x10IncidentResponse\x12\x12\n\x05error\x18\x01 \x01(\tH\x00\x88\x01\x01B\x08\n\x06_error"g\n\x10DriveSessionInfo\x12.\n\ntime_stamp\x18\x01 \x01(\x0b2\x1a.google.protobuf.Timestamp\x12\x12\n\ndrive_name\x18\x02 \x01(\t\x12\x0f\n\x07version\x18\x03 \x01(\r"\xd4\x06\n\x04Gnss\x12.\n\ntime_stamp\x18\x01 \x01(\x0b2\x1a.google.protobuf.Timestamp\x12\x19\n\x11session_timestamp\x18\x11 \x01(\x04\x12+\n\x06status\x18\x02 \x01(\x0e2\x1b.intercom.Gnss.NavSatStatus\x12*\n\x07service\x18\x03 \x01(\x0e2\x19.intercom.Gnss.NavService\x12\x10\n\x08latitude\x18\x04 \x01(\x01\x12\x11\n\tlongitude\x18\x05 \x01(\x01\x12\x10\n\x08altitude\x18\x06 \x01(\x01\x12\x0f\n\x07heading\x18\x07 \x01(\x01\x12\x15\n\rheading_error\x18\x08 \x01(\x01\x12\r\n\x05speed\x18\t \x01(\x01\x12\x13\n\x0bspeed_error\x18\n \x01(\x01\x12\x1d\n\x15has_highprecision_loc\x18\x0b \x01(\x08\x12\x17\n\x0fhp_loc_latitude\x18\x0c \x01(\x01\x12\x18\n\x10hp_loc_longitude\x18\r \x01(\x01\x12\x17\n\x0fhp_loc_altitude\x18\x0e \x01(\x01\x12\x1b\n\x13position_covariance\x18\x0f \x03(\x01\x12:\n\x18position_covariance_type\x18\x10 \x01(\x0e2\x18.intercom.Gnss.NavSatFix"d\n\x0cNavSatStatus\x12\x0e\n\nSTATUS_FIX\x10\x00\x12\x13\n\x0fSTATUS_SBAS_FIX\x10\x01\x12\x13\n\x0fSTATUS_GBAS_FIX\x10\x02\x12\x1a\n\rSTATUS_NO_FIX\x10\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01"o\n\nNavService\x12\x10\n\x0cSERVICE_NONE\x10\x00\x12\x0f\n\x0bSERVICE_GPS\x10\x01\x12\x14\n\x10SERVICE_GLOSNASS\x10\x02\x12\x13\n\x0fSERVICE_COMPASS\x10\x04\x12\x13\n\x0fSERVICE_GALILEO\x10\x08"\x89\x01\n\tNavSatFix\x12\x1b\n\x17COVARIANCE_TYPE_UNKNOWN\x10\x00\x12 \n\x1cCOVARIANCE_TYPE_APPROXIMATED\x10\x01\x12"\n\x1eCOVARIANCE_TYPE_DIAGONAL_KNOWN\x10\x02\x12\x19\n\x15COVARIANCE_TYPE_KNOWN\x10\x03"\xa4\x02\n\rImageMetadata\x12.\n\ntime_stamp\x18\x01 \x01(\x0b2\x1a.google.protobuf.Timestamp\x12\x19\n\x11session_timestamp\x18\x0c \x01(\x04\x12\x13\n\x0bcamera_name\x18\x02 \x01(\t\x12\x10\n\x08encoding\x18\x03 \x01(\t\x12\x15\n\rexposure_time\x18\x04 \x01(\x01\x12\x11\n\tframe_idx\x18\x05 \x01(\x04\x12\x15\n\rgain_level_db\x18\x06 \x01(\x01\x12\x1a\n\x12gain_level_digital\x18\x07 \x01(\x01\x12\x0e\n\x06height\x18\x08 \x01(\r\x12\r\n\x05width\x18\t \x01(\r\x12\x11\n\tscene_lux\x18\n \x01(\x01\x12\x12\n\nsensor_iso\x18\x0b \x01(\r2\xb3\x02\n\rSensorService\x12J\n\x0eStartRecording\x12\x1f.intercom.StartRecordingRequest\x1a\x17.intercom.SessionStatus\x12N\n\x10StartRecordingV2\x12!.intercom.StartRecordingRequestV2\x1a\x17.intercom.SessionStatus\x12D\n\x12PrioritizeIncident\x12\x12.intercom.Incident\x1a\x1a.intercom.IncidentResponse\x12@\n\rStopRecording\x12\x16.google.protobuf.Empty\x1a\x17.intercom.SessionStatusB\x06Z\x04.;pbb\x06proto3')
_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'sensor_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    _globals['DESCRIPTOR']._loaded_options = None
    _globals['DESCRIPTOR']._serialized_options = b'Z\x04.;pb'
    _globals['_STARTRECORDINGREQUEST']._serialized_start = 88
    _globals['_STARTRECORDINGREQUEST']._serialized_end = 203
    _globals['_STARTRECORDINGREQUESTV2']._serialized_start = 205
    _globals['_STARTRECORDINGREQUESTV2']._serialized_end = 275
    _globals['_SESSIONSTATUS']._serialized_start = 277
    _globals['_SESSIONSTATUS']._serialized_end = 382
    _globals['_INCIDENT']._serialized_start = 384
    _globals['_INCIDENT']._serialized_end = 479
    _globals['_INCIDENTRESPONSE']._serialized_start = 481
    _globals['_INCIDENTRESPONSE']._serialized_end = 529
    _globals['_DRIVESESSIONINFO']._serialized_start = 531
    _globals['_DRIVESESSIONINFO']._serialized_end = 634
    _globals['_GNSS']._serialized_start = 637
    _globals['_GNSS']._serialized_end = 1489
    _globals['_GNSS_NAVSATSTATUS']._serialized_start = 1136
    _globals['_GNSS_NAVSATSTATUS']._serialized_end = 1236
    _globals['_GNSS_NAVSERVICE']._serialized_start = 1238
    _globals['_GNSS_NAVSERVICE']._serialized_end = 1349
    _globals['_GNSS_NAVSATFIX']._serialized_start = 1352
    _globals['_GNSS_NAVSATFIX']._serialized_end = 1489
    _globals['_IMAGEMETADATA']._serialized_start = 1492
    _globals['_IMAGEMETADATA']._serialized_end = 1784
    _globals['_SENSORSERVICE']._serialized_start = 1787
    _globals['_SENSORSERVICE']._serialized_end = 2094
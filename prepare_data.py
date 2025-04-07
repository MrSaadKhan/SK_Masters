import random
import math
import clean_data, group_data, number_to_words

def prepare_data(file_path, group_option=0, time_group=0, num2word_option=0):
    
    data_map = {
        "flowStartMilliseconds"                     : "flow Start Milliseconds",
        "flowEndMilliseconds"                       : "flow End Milliseconds",
        "flowDurationMilliseconds"                  : "flow Duration Milliseconds",
        "reverseFlowDeltaMilliseconds"              : "reverse Flow Delta Milliseconds",
        "protocolIdentifier"                        : "protocol Identifier",
        "sourceIPv4Address"                         : "source IPv4 Address",
        "sourceTransportPort"                       : "source Transport Port",
        "packetTotalCount"                          : "packet Total Count",
        "octetTotalCount"                           : "octet Total Count",
        "flowAttributes"                            : "flow Attributes",
        "sourceMacAddress"                          : "source Mac Address",
        "destinationIPv4Address"                    : "destination IPv4 Address",
        "destinationTransportPort"                  : "destination Transport Port",
        "reversePacketTotalCount"                   : "reverse Packet Total Count",
        "reverseOctetTotalCount"                    : "reverse Octet Total Count",
        "reverseFlowAttributes"                     : "reverse Flow Attributes",
        "destinationMacAddress"                     : "destination Mac Address",
        "initialTCPFlags"                           : "initial TCP Flags",
        "unionTCPFlags"                             : "union TCP Flags",
        "reverseInitialTCPFlags"                    : "reverse Initial TCP Flags",
        "reverseUnionTCPFlags"                      : "reverse Union TCP Flags",
        "tcpSequenceNumber"                         : "tcp Sequence Number",
        "reverseTcpSequenceNumber"                  : "reverse Tcp Sequence Number",
        "ingressInterface"                          : "ingress Interface",
        "egressInterface"                           : "egress Interface",
        "vlanId"                                    : "vlan Id",
        "silkAppLabel"                              : "silkApp Label",
        "ipClassOfService"                          : "ip Class Of Service",
        "flowEndReason"                             : "flow End Reason",
        "collectorName"                             : "collector Name",
        "observationDomainId"                       : "observation Domain Id",
        "tcpUrgTotalCount"                          : "tcp Urgent Total Count",
        "smallPacketCount"                          : "small Packet Count",
        "nonEmptyPacketCount"                       : "non Empty Packet Count",
        "dataByteCount"                             : "data Byte Count",
        "averageInterarrivalTime"                   : "average Interarrival Time",
        "firstNonEmptyPacketSize"                   : "first Non Empty Packet Size",
        "largePacketCount"                          : "large Packet Count",
        "maxPacketSize"                             : "maximum Packet Size",
        "firstEightNonEmptyPacketDirections"        : "first Eight Non Empty Packet Directions",
        "standardDeviationPayloadLength"            : "standard Deviation Payload Length",
        "standardDeviationInterarrivalTime"         : "standard Deviation Interarrival Time",
        "bytesPerPacket"                            : "bytes Per Packet",
        "reverseTcpUrgTotalCount"                   : "reverse Tcp Urgent Total Count",
        "reverseSmallPacketCount"                   : "reverse Small Packet Count",
        "reverseNonEmptyPacketCount"                : "reverse Non Empty Packet Count",
        "reverseDataByteCount"                      : "reverse Data Byte Count",
        "reverseAverageInterarrivalTime"            : "reverse Average Interarrival Time",
        "reverseFirstNonEmptyPacketSize"            : "reverse First Non Empty Packet Size",
        "reverseLargePacketCount"                   : "reverse Large Packet Count",
        "reverseMaxPacketSize"                      : "reverse Maximum Packet Size",
        "reverseStandardDeviationPayloadLength"     : "reverse Standard Deviation Payload Length",
        "reverseStandardDeviationInterarrivalTime"  : "reverse Standard Deviation Interarrival Time",
        "reverseBytesPerPacket"                     : "reverse Bytes Per Packet"
    }

    
    def replace_words_in_dict_list(data_list, word_map):
        # print(data_list)
        updated_list = []
        for data_dict in data_list:
            new_dict = {}
            for key, value in data_dict.items():
                # Replace the key with the mapped value if found, else keep the original key
                new_key = word_map.get(key, key)
                new_dict[new_key] = value
            updated_list.append(new_dict)
        return updated_list


    num_elements = []
    data = []

    # Clean the data for the single file
    temp, temp1 = clean_data.clean_data(file_path)
    data.append(temp)  # Append the cleaned data to the list
    num_elements.append(temp1)

    def split_list(lst, split_index):
        return lst[:split_index], lst[split_index:]

    def flatten(data):
        flattened_data = []
        for sublist in data:
            flattened_data.extend(sublist)
        return flattened_data

    flattened_data = flatten(data)
    dev1, _ = split_list(flattened_data, num_elements[0])  # Split the list (second part is unused)

    dev1 = replace_words_in_dict_list(dev1, data_map)
    del data, flattened_data

    def random_split(lst, n):
        selected_indices = set(random.sample(range(len(lst)), n))
        selected_items = [lst[i] for i in selected_indices]
        remaining_items = [lst[i] for i in range(len(lst)) if i not in selected_indices]
        return selected_items, remaining_items

    def apply_group_data(dataset):
        if len(dataset) == 0:
            return [], []
        else:
            unseen, seen = random_split(dataset, math.floor(0.3 * len(dataset)))
            return group_data.group_data(unseen, time_group), group_data.group_data(seen, time_group)

    if group_option == 1:
        dev1_unseen, dev1_seen = apply_group_data(dev1)
    else:
        dev1_unseen, dev1_seen = random_split(dev1, math.floor(0.3 * num_elements[0]))

    del dev1

    if num2word_option == 1:
        dev1_seen = number_to_words.convert_numericals_to_words(dev1_seen)
        dev1_unseen = number_to_words.convert_numericals_to_words(dev1_unseen)

    print('\033[92mData prepared successfully ✔\033[0m')
    return dev1_seen, dev1_unseen


# if __name__ == "__main__":
#     print(prepare_data("/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix/planex_smacam_pantilt.json"))
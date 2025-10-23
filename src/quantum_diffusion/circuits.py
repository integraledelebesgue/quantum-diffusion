import pennylane as qml
import qw_map


def sampling_qnode(wires: int, num_repeats: int) -> qml.QNode:
    def __circuit(inp_image, weights):
        qml.QubitStateVector(inp_image, wires=range(wires))

        for _ in range(num_repeats):
            qml.StronglyEntanglingLayers(qw_map.tanh(weights), wires=range(wires))

        return qml.state()

    return qml.QNode(
        func=__circuit,
        device=qml.device("default.qubit", wires=wires),
        interface="torch",
        diff_method=None,
    )


def sampling_qnode_with_swap(
    wires: int,
    num_repeats: int,
    has_reuploads: bool,
) -> qml.QNode:
    """
    Create a new qnode with more wires.
    For each num_repeat, an additional wire is added to inject the label rotation.
    THIS IS JUST A WORKAROUNG UNTIL PENNYLANE SUPPORTS THE RESET GATE.
    """

    def __circuit(inp_image, weights, label):
        qml.QubitStateVector(inp_image, wires=range(wires - 1))

        for rep in range(num_repeats):
            if has_reuploads:
                for w in weights:
                    if label is not None:
                        qml.RX(label, wires=wires - 1)

                    qml.StronglyEntanglingLayers(qw_map.tanh(w), wires=range(wires))
            else:
                if label is not None:
                    qml.RX(label, wires=wires - 1)

                qml.StronglyEntanglingLayers(qw_map.tanh(weights), wires=range(wires))

            qml.SWAP(wires=[wires - 1, wires + rep])

        return qml.probs(wires=range(wires))

    return qml.QNode(
        func=__circuit,
        device=qml.device("default.qubit", wires=wires + num_repeats),
        interface="torch",
        diff_method="backprop",
    )

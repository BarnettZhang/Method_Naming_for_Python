import zmq
import sys


if __name__ == "__main__":
    with open('isFirstExecution.txt', 'w') as f:
        f.write('true')
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    socket.send(str.encode(sys.argv[1]))
    message = socket.recv().decode()
    print(message)
    sys.stdout.flush()
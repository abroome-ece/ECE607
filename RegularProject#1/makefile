all: task1 task2 task3

.venv:
	py -m venv .venv
	.venv\Scripts\python -m pip install --upgrade pip
	.venv\Scripts\pip install numpy matplotlib pandas scikit-learn

task1: .venv
	.venv\Scripts\python "task1.py"

task2: .venv
	.venv\Scripts\python "task2.py"

task3: .venv
	.venv\Scripts\python "task3.py"

clean:
	rmdir /s /q .venv
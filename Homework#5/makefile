.venv:
	py -m venv .venv
	.venv\Scripts\python -m pip install --upgrade pip
	.venv\Scripts\pip install numpy matplotlib scipy pandas scikit-learn seaborn

problem1: .venv
	.venv\Scripts\python "problem1.py"

problem2: .venv
	.venv\Scripts\python "problem2.py"

problem3: .venv
	.venv\Scripts\python "problem3.py"

clean:
	rmdir /s /q .venv
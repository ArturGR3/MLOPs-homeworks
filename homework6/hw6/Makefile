.PHONY: quality_checks tests integration_tests run

quality_checks:
	isort .
	black .
	pylint .

tests: 
	pytest tests/

integration_tests:
	bash run.sh

# The 'run' target is used to execute your application.
run: quality_checks tests integration_tests
	echo "Running the application"

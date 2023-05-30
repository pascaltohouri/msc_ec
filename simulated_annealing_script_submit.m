% this script takes a transposed csv file of 'ulysis22TSP.csv' with columns
% [ID,X,Y]

% Start timing the algorithm
tic;

% open the CSV file
fid = fopen('ulysis22TSP.csv', 'r');

% skip the first line containing headers
fgets(fid);

% scan the remaining lines into a cell array called 'data'
data = textscan(fid, '%f%f%f', 'Delimiter', ',');

% close the file
fclose(fid);

% convert the cell array to a matrix 'nodes' and remove the first column
nodes = cell2mat(data(2:3));

% define the number of nodes
num_nodes = size(nodes, 1);

% define the number of iterations to run and filename to save results,
% 'simulated_annealing_results.csv'
n = 30;
filename = 'simulated_annealing_results.csv';

% set the initial temperature, cooling rate, and number of iterations
initial_temperature = 20;
cooling_rate = 0.95;
num_iterations = 20000;

% define a function to calculate the Euclidean distance between two nodes
euclidean_distance = @(a, b) sqrt(sum((nodes(a, :) - nodes(b, :)).^2));

% define the objective function to be optimized
objective = @(x) sum(arrayfun(@(i) euclidean_distance(x(i), x(mod(i,num_nodes)+1)), 1:num_nodes));

% create a matrix to store results 'results'
simulated_annealing_results_matrix = zeros(30, num_nodes+1);

for j = 1:30
    % generate a random initial hamiltonian cycle
    current_solution = randperm(num_nodes);

    % calculate the objective value of the initial solution
    current_obj_value = objective(current_solution);

    % start the simulated annealing algorithm
    temperature = initial_temperature;
    for i = 1:num_iterations
        % generate a neighboring solution by swapping two random nodes in the hamiltonian cycle
        neighbor_solution = current_solution;
        idx = randperm(num_nodes, 2);
        neighbor_solution(idx) = neighbor_solution(flip(idx));

        % calculate the objective value of the neighboring solution
        neighbor_obj_value = objective(neighbor_solution);

        % calculate the energy difference between the current and neighboring solutions
        energy_diff = neighbor_obj_value - current_obj_value;

        % decide whether to accept the neighboring solution or not
        if energy_diff < 0 || rand() < exp(-energy_diff/temperature)
            current_solution = neighbor_solution;
            current_obj_value = neighbor_obj_value;

        end
        
        % decrease the temperature
        temperature = cooling_rate * temperature;
 
    end

    % Store the results in the matrix
    simulated_annealing_results_matrix(j, 1:num_nodes) = current_solution;
    simulated_annealing_results_matrix(j, end) = current_obj_value;
end

% Save the results to a CSV file
csvwrite(filename, simulated_annealing_results_matrix);

disp(simulated_annealing_results_matrix)

% Stop timing and get the elapsed time
elapsed_time = toc;
fprintf('Elapsed time: %f seconds\n', elapsed_time);

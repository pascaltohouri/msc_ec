% Start timing the algorithm
tic;

% load the node coordinates from the ulysis22TSP.csv file
data = csvread('ulysis22TSP.csv', 1, 0); % start reading from row 2
nodes = data(:,2:3);

% set the genetic algorithm parameters
runs = 30;
pop_size = 100;
selection_rate = 0.2;
crossover_rate = 1;
mutation_rate = 0.02;
generations = 123;

% calculate the distance matrix between nodes
dist_matrix = pdist(nodes);
dist_matrix = squareform(dist_matrix);

% iterate for all runs
for j = 1:runs
    
    % generate initial population
    population = zeros(pop_size, length(nodes));
    for i = 1:pop_size
        population(i,:) = randperm(length(nodes));
    end
    % iterate over generations
    for gen = 1:generations
        % evaluate fitness of each individual in population
        fitness_scores = zeros(1, pop_size);
        for i = 1:pop_size
            fitness_scores(i) = fitness(population(i,:), nodes);
        end
        
        % sort population by fitness score
        [~, idx] = sort(fitness_scores);
        population = population(idx,:);
        
        % keep track of best individual and its fitness score
        best_individual = population(1,:);
        best_fitness_score = fitness_scores(1);
        
        % select parents using tournament selection
        parents = tournament(fitness_scores, population, selection_rate);

        % perform crossover and mutation to generate offspring
        offspring = zeros(size(population));
        for i = 1:2:pop_size
            if length(unique(parents(i,:))) == length(parents(i,:)) && length(unique(parents(i+1,:))) == length(parents(i+1,:))
                if rand < crossover_rate
                    [offspring1, offspring2] = crossover(parents(i,:), parents(i+1,:));
                else
                offspring1 = parents(i,:);
                offspring2 = parents(i+1,:);
                end

                % mutate offspring
                if rand < mutation_rate
                    offspring1 = mutation(offspring1);
                end
                if rand < mutation_rate
                    offspring2 = mutation(offspring2);
                end

                % add offspring to population
                population(end-1,:) = offspring1;
                population(end,:) = offspring2;
            end 
        end

        % evaluate fitness of each individual in population
        fitness_scores = zeros(1, pop_size);
        for i = 1:pop_size
            fitness_scores(i) = fitness(population(i,:), nodes);
        end
        
        % sort population by fitness score
        [~, idx] = sort(fitness_scores);
        population = population(idx,:);
        
        % keep track of best individual and its fitness score
        if fitness_scores(1) < best_fitness_score
            best_individual = population(1,:);
            best_fitness_score = fitness_scores(1);
        end
        
        % print current best fitness score
        % fprintf('Generation %d: best fitness score = %f\n', gen, best_fitness_score);
    end
    % print final best individual and its fitness score
    fprintf('Run %d: best individual = [%s], best fitness score = %f\n', j, num2str(best_individual), best_fitness_score);

    % save best individual and its fitness score in results matrix
    % genetic_algorithm_results_matrix(j,1:length(nodes)) = best_individual;
    % genetic_algorithm_results_matrix(j,length(nodes)+1) = best_fitness_score;
    
end

% filename = 'genetic_algorithm_results.csv';
% csvwrite(filename, genetic_algorithm_results_matrix);

% Stop timing and get the elapsed time
elapsed_time = toc;
fprintf('Elapsed time: %f seconds\n', elapsed_time);

% define the fitness function to calculate the total distance of a given tour
function dist = fitness(tour, nodes)
    % calculate the total distance of the tour
    dist = 0;
    for i = 1:length(tour)-1
        dist = dist + norm(nodes(tour(i),:) - nodes(tour(i+1),:));
    end
    % add the distance from the last node back to the first node
    dist = dist + norm(nodes(tour(end),:) - nodes(tour(1),:));
end


% define the tournament selection function to select parents from the population
function parents = tournament(fitness_scores, population, selection_rate)
    % randomly select two individuals for each parent
    tournament = randperm(size(population,1), size(population,1)*selection_rate);
    tournament = reshape(tournament, [2,size(population,1)*(0.5*selection_rate)])';
    % choose the better individual as the parent
    better = fitness_scores(tournament(:,1)) < fitness_scores(tournament(:,2));
    parents = zeros(size(population));
    parents(better,:) = population(tournament(better,1),:);
    parents(~better,:) = population(tournament(~better,2),:);
end



function [offspring1, offspring2] = crossover(parent1, parent2)
    % Check inputs
    if ~isvector(parent1) || ~isvector(parent2)
        disp('Parents must be vectors')
    end
    if length(parent1) ~= length(parent2)
        disp('Parents must have the same length')
    end
    if length(unique(parent1)) ~= length(parent1) || length(unique(parent2)) ~= length(parent2)
        disp('Parents must not contain repeated elements')
    end
    
    % Randomly select a crossover point
    point = randi(length(parent1));
    
    % Create the two offspring by swapping segments between the parents
    offspring1 = zeros(1,length(parent1));
    offspring2 = zeros(1,length(parent1));
    offspring1(1:point) = parent1(1:point);
    offspring2(1:point) = parent2(1:point);

    % Obtain parent 2 nodes, not in parent 1 until x-over (temp1)
    set1 = setdiff(parent2, parent1(1:point), 'stable');
    % Obtain parent 1 nodes, not in parent 2 until x-over (temp2)
    set2 = setdiff(parent1, parent2(1:point), 'stable');

    % Assign remaining nodes to offspring
    offspring1(point+1:end) = set1;
    offspring2(point+1:end) = set2;
end

% define the mutation function to introduce small changes to a tour
function tour = mutation(tour)
    % randomly swap two nodes in the tour
    idx = randperm(length(tour), 2);
    tour([idx(1) idx(2)]) = tour([idx(2) idx(1)]);
end

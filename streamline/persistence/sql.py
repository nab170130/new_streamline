import time
import sqlite3

def _get_db_connection(file_loc):
    
    db_connection = sqlite3.connect(file_loc, timeout=30)
    db_connection.execute("PRAGMA foreign_keys = 1")
    return db_connection

def create_db(file_loc):
    
    db_connection = _get_db_connection(file_loc)
    
    try:
        with db_connection:
            db_connection.execute("BEGIN ")
            db_connection.execute("CREATE TABLE Experiment (experiment_id INT, \
                                                            dataset_name TEXT, \
                                                            model_name TEXT, \
                                                            limited_mem INT, \
                                                            arrival_pattern TEXT, \
                                                            CONSTRAINT pkey_experiment PRIMARY KEY (experiment_id), \
                                                            UNIQUE(dataset_name, model_name, limited_mem, arrival_pattern) \
                                                            )")
                
            db_connection.execute("CREATE TABLE ALRun (experiment_id INT, \
                                                       run_id INT, \
                                                       run_number INT, \
                                                       training_loop TEXT, \
                                                       al_method TEXT, \
                                                       budget INT, \
                                                       init_task_size INT, \
                                                       unl_buffer_size INT, \
                                                       CONSTRAINT pkey_alrun PRIMARY KEY (experiment_id, run_id), \
                                                       CONSTRAINT fkey_alrun_experiment FOREIGN KEY (experiment_id) REFERENCES Experiment(experiment_id), \
                                                       UNIQUE(experiment_id, run_number, training_loop, al_method, budget, init_task_size, unl_buffer_size) \
                                                       )")

            db_connection.execute("CREATE TABLE ALRound (experiment_id INT, \
                                                         run_id INT, \
                                                         round_id INT, \
                                                         current_epoch INT, \
                                                         dataset_split_path TEXT, \
                                                         model_weight_path TEXT, \
                                                         opt_state_path TEXT, \
                                                         lr_state_path TEXT, \
                                                         completed_unix_time INT, \
                                                         CONSTRAINT pkey_alround PRIMARY KEY (experiment_id, run_id, round_id), \
                                                         CONSTRAINT fkey_alround_alrun FOREIGN KEY (experiment_id, run_id) \
                                                                                       REFERENCES ALRun(experiment_id, run_id) \
                                                         )")
            
            db_connection.execute("CREATE TABLE ALParam (experiment_id INT, \
                                                         run_id INT, \
                                                         round_id INT, \
                                                         name TEXT, \
                                                         value REAL, \
                                                         CONSTRAINT pkey_alparam PRIMARY KEY (experiment_id, run_id, round_id, name), \
                                                         CONSTRAINT fkey_alparam_alround FOREIGN KEY (experiment_id, run_id, round_id) \
                                                                                         REFERENCES ALRound(experiment_id, run_id, round_id) ON DELETE CASCADE \
                                                         )")
                
            db_connection.execute("CREATE TABLE StopCriterion (experiment_id INT, \
                                                               run_id INT, \
                                                               round_id INT, \
                                                               cname TEXT, \
                                                               pname TEXT, \
                                                               value REAL, \
                                                               CONSTRAINT pkey_stopcrit PRIMARY KEY (experiment_id, run_id, round_id, cname, pname), \
                                                               CONSTRAINT fkey_stopcrit_alround FOREIGN KEY (experiment_id, run_id, round_id) \
                                                                                                REFERENCES ALRound(experiment_id, run_id, round_id) ON DELETE CASCADE \
                                                               )")
            
            db_connection.execute("CREATE TABLE Metric (experiment_id INT, \
                                                        run_id INT, \
                                                        round_id INT, \
                                                        name TEXT, \
                                                        eval_dataset TEXT, \
                                                        value REAL, \
                                                        computed_unix_time INT, \
                                                        CONSTRAINT pkey_metric PRIMARY KEY (experiment_id, run_id, round_id, name, eval_dataset), \
                                                        CONSTRAINT fkey_metric_alround FOREIGN KEY (experiment_id, run_id, round_id) \
                                                                                       REFERENCES ALRound(experiment_id, run_id, round_id) ON DELETE CASCADE \
                                                        )")
                
            db_connection.execute("CREATE TABLE Augment (experiment_id INT, \
                                                         run_id INT, \
                                                         round_id INT, \
                                                         is_train INT, \
                                                         aug_order INT, \
                                                         descriptor TEXT, \
                                                         CONSTRAINT pkey_augment PRIMARY KEY (experiment_id, run_id, round_id, is_train, aug_order), \
                                                         CONSTRAINT fkey_augment_alround FOREIGN KEY (experiment_id, run_id, round_id) \
                                                                                         REFERENCES ALRound(experiment_id, run_id, round_id) ON DELETE CASCADE \
                                                         )")
    except sqlite3.OperationalError as err:
        print(F"Could not construct database or database already exists: {err}")
        
    db_connection.close()


def get_experiment_id(cursor, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern):

    # There needs to be an experiment ID if one does not exist for the passed combination. Here, we shall see if there is such an experiment ID.
    # As there is a uniqueness constraint on (dataset_name, model_name, limited_mem, arrival_pattern), we can use such tuples as a superkey.
    cursor.execute("SELECT experiment_id FROM Experiment WHERE dataset_name = ? AND model_name = ? AND limited_mem = ? AND arrival_pattern = ?",
                    (train_dataset_name, model_architecture_name, limited_mem, arrival_pattern))
    experiment_id = cursor.fetchone()

    if experiment_id is not None:
        experiment_id = experiment_id[0]
            
    # Return the experiment_id, which could be nothing.
    return experiment_id


def create_experiment_id(cursor):

    # To create a unique experiment ID, we simply take the largest experiment ID and increment it by 1.
    cursor.execute("SELECT experiment_id FROM Experiment ORDER BY experiment_id DESC LIMIT 1")
    experiment_id = cursor.fetchone()

    if experiment_id is None:
        experiment_id = 100
    else:
        experiment_id = experiment_id[0] + 1

    return experiment_id


def get_run_id(cursor, experiment_id, run_number, training_loop, al_method, budget, init_task_size, unl_buffer_size):

    # There needs to be an experiment ID if one does not exist for the passed combination. Here, we shall see if there is such an experiment ID.
    # As there is a uniqueness constraint on (dataset_name, model_name, limited_mem, arrival_pattern), we can use such tuples as a superkey.
    cursor.execute("SELECT run_id FROM ALRun WHERE experiment_id = ? AND run_number = ? AND training_loop = ? AND al_method = ? AND budget = ? AND init_task_size = ? AND unl_buffer_size = ?",
                    (experiment_id, run_number, training_loop, al_method, budget, init_task_size, unl_buffer_size))
    run_id = cursor.fetchone()

    if run_id is not None:
        run_id = run_id[0]      

    # Return the run_id, which could be nothing.
    return run_id


def create_run_id(cursor, experiment_id):

    # To create a unique run ID, we simply take the largest run ID corresponding to the passed experiment and increment it by 1.
    cursor.execute("SELECT run_id FROM ALRun WHERE experiment_id = ? ORDER BY run_id DESC LIMIT 1", (experiment_id,))
    run_id = cursor.fetchone()

    if run_id is None:
        run_id = 100
    else:
        run_id = run_id[0] + 1

    return run_id


def add_al_round(db_loc,
                 train_dataset_name, 
                 model_architecture_name, 
                 limited_mem,
                 arrival_pattern,
                 run_number, 
                 training_loop,
                 al_method_name, 
                 al_budget, 
                 init_task_size,
                 unl_buffer_size,
                 round_number, 
                 current_epoch,
                 dataset_split_path, 
                 model_weight_path,
                 optimizer_state_path,
                 lr_scheduler_state_path,
                 completion_unix_time,
                 al_params,
                 train_augmentations,
                 test_augmentations,
                 stop_criteria,
                 metrics,
                 attempt=0):
    
    # Start the main transaction
    db_connection = _get_db_connection(db_loc)

    try:
        with db_connection:
            
            cursor = db_connection.cursor()
            cursor.execute("BEGIN ")

            # First, get the experiment id (if it exists). If it does not exist, then create a new id that
            # does not exist in the DB. If it does, then get the element from the tuple. Do the same in procuring the run id. The round id IS
            # the round number.
            experiment_id = get_experiment_id(cursor, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern)
            if experiment_id is None:
                experiment_id = create_experiment_id(cursor)

            run_id = get_run_id(cursor, experiment_id, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size)
            if run_id is None:
                run_id = create_run_id(cursor, experiment_id)

            # Ensure that there's an experiment encompassing this round
            cursor.execute("INSERT INTO Experiment VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING", (experiment_id, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern))
                
            # Ensure that there's an AL run encompassing this round
            cursor.execute("INSERT INTO ALRun VALUES (?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING", (experiment_id, run_id, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size))
                
            # Since we're adding another round, purge any existing information currently matching the round to be added (and all subsequent rounds that might exist). 
            # We do not want stale data.
            cursor.execute("DELETE FROM ALRound WHERE experiment_id = ? AND run_id = ? AND round_id >= ?", (experiment_id, run_id, round_number))

            # Before adding, enforce the constraint that the AL round number must be no greater than two past the max currently stored round number.
            # This will allow us to ensure rounds are built up sequentially.
            cursor.execute("SELECT MAX(round_id) FROM ALRound WHERE experiment_id = ? AND run_id = ?", (experiment_id, run_id))
            max_round = cursor.fetchone()[0]
            
            if max_round is None:
                max_round = -1
            else:
                max_round = max_round
            if round_number >= max_round + 2:
                raise sqlite3.IntegrityError(F"Attempted to insert round {round_number} when maximum round is {max_round}")
            
            # Begin inserting the new record(s) into the appropriate tables, starting with the AL round
            cursor.execute("INSERT INTO ALRound VALUES (?,?,?,?,?,?,?,?,?)", (experiment_id, run_id, round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_scheduler_state_path, completion_unix_time))
                                                                                                                
            # Add augmentations into database
            for order_number, transform_descriptor in enumerate(train_augmentations):
                cursor.execute("INSERT INTO Augment VALUES (?,?,?,?,?,?)",
                               (experiment_id, run_id, round_number, 1, order_number, transform_descriptor))
            
            for order_number, transform_descriptor in enumerate(test_augmentations):
                cursor.execute("INSERT INTO Augment VALUES (?,?,?,?,?,?)",
                               (experiment_id, run_id, round_number, 0, order_number, transform_descriptor))
            
            # Add stop criteria into database
            for cname, pname, value in stop_criteria:
                cursor.execute("INSERT INTO StopCriterion VALUES (?,?,?,?,?,?)",
                               (experiment_id, run_id, round_number, cname, pname, value))
            
            # Add metrics into database
            for name, eval_dataset, value, computed_time in metrics:
                cursor.execute("INSERT INTO Metric VALUES (?,?,?,?,?,?,?)",
                               (experiment_id, run_id, round_number, name, eval_dataset, value, computed_time))
            
            # Add AL parameters into database
            for name, value in al_params:
                cursor.execute("INSERT INTO ALParam VALUES (?,?,?,?,?)",
                               (experiment_id, run_id, round_number, name, value))
            
    except sqlite3.OperationalError as err:

        # Instead of wasting the attempt in the event of a locked DB, retry adding the AL round. If we've tried enough, though, then stop attempting.
        if attempt == 5:
            raise err
        else:
            time.sleep(1)
            add_al_round(db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size,
                            unl_buffer_size, round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_scheduler_state_path, completion_unix_time,
                            al_params, train_augmentations, test_augmentations, stop_criteria, metrics, attempt + 1)
    except Exception as err:
        raise err
    finally:
        db_connection.close()


def add_metric_time_value_to_run(db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, budget, init_task_size, unl_buffer_size, round_num, metric_name, eval_dataset, metric_value, time_value, attempt=0):
    
    db_connection = _get_db_connection(db_loc)

    try:
        with db_connection:
            db_cursor = db_connection.cursor()
            db_cursor.execute("BEGIN ")

            # First, get the experiment id and the run id (if they exist). If either does not exist, raise
            # an error
            experiment_id   = get_experiment_id(db_cursor, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern)
            if experiment_id is None:   raise ValueError("No matching experiment in DB")
            run_id          = get_run_id(db_cursor, experiment_id, run_number, training_loop, al_method_name, budget, init_task_size, unl_buffer_size)
            if run_id is None:          raise ValueError("No matching run in DB")

            # Insert the metric values into the database, overwriting existing metric values
            db_cursor.execute("INSERT INTO Metric VALUES (?,?,?,?,?,?,?) ON CONFLICT DO UPDATE SET value=excluded.value, computed_unix_time=excluded.computed_unix_time",
                              (experiment_id, run_id, round_num, metric_name, eval_dataset, metric_value, time_value))
        
    except Exception as err:
        
        # Instead of wasting the attempt in the event of a locked DB, retry adding the metric. If we've tried enough, though, then stop attempting.
        if attempt == 5:
            raise err
        else:
            time.sleep(1)
            add_metric_time_value_to_run(db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, budget, init_task_size, unl_buffer_size, round_num, metric_name, eval_dataset, metric_value, time_value, attempt + 1)
        raise err
    finally:
        db_connection.close()


def get_valid_metric_values_computed_times_over_all_runs(db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, training_loop, al_method_name, 
                                                         budget, init_task_size, unl_buffer_size, metric_name, eval_dataset):

    db_connection = _get_db_connection(db_loc)
    
    try:
        with db_connection:
            db_cursor = db_connection.cursor()
            db_cursor.execute("BEGIN ")
            
            # First, get the experiment id. If it does not exist, raise an error
            experiment_id   = get_experiment_id(db_cursor, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern)
            if experiment_id is None:   raise ValueError("No matching experiment in DB")

            # We want only those runs that align in ALRun:
            #   WHERE experiment_id = ? AND training_loop = ? AND al_method = ? AND budget = ? AND init_task_size = ? AND unl_buffer_size = ?
            #
            # And we only want those metric values that correspond to the named metric + eval dataset:
            #   WHERE name = ? AND eval_dataset = ?
            #
            # We additionally need one more join to get the completed unix time
            #
            # Once joined, we specifically want those tuples whose experiments finished BEFORE the metric was computed AND are done training AND HAVE computed values
            # This gives the below query:
            db_cursor.execute("SELECT run_number, round_id, value, computed_unix_time \
                              FROM  (SELECT *\
                                     FROM ALRun\
                                     WHERE experiment_id = ? AND training_loop = ? AND al_method = ? AND budget = ? AND init_task_size = ? AND unl_buffer_size = ?)\
                              NATURAL JOIN ALRound\
                              NATURAL JOIN\
                                    (SELECT *\
                                     FROM Metric\
                                     WHERE name = ? AND eval_dataset = ?)\
                              WHERE (completed_unix_time < computed_unix_time) AND (computed_unix_time IS NOT NULL) AND (current_epoch < 0)",
                              (experiment_id, training_loop, al_method_name, budget, init_task_size, unl_buffer_size, metric_name, eval_dataset))
            run_round_value_tuples = db_cursor.fetchall()
            
            # Regroup values into a list of lists. The outer list contains lists corresponding to each run.
            # Each inner list contains the metric value at each round.
            run_numbers = sorted(list(set([x[0] for x in run_round_value_tuples])))
            
            run_metric_value_lists = []
            run_metric_time_lists = []
            for run_number in run_numbers:
                relevant_tuples = [run_round_value_tuple for run_round_value_tuple in run_round_value_tuples if run_round_value_tuple[0] == run_number]
                max_round = max(relevant_tuples, key=lambda x:x[1])[1]
                metric_value_list = [None for x in range(max_round + 1)]
                metric_time_list = [None for x in range(max_round + 1)]
                for _, round_number, metric_value, time_value in relevant_tuples:
                    metric_value_list[round_number] = metric_value
                    metric_time_list[round_number] = time_value
                run_metric_value_lists.append(metric_value_list)
                run_metric_time_lists.append(metric_time_list)

    except Exception as err:
        print("Error in retrieving metric info")
        raise err
    finally:
        db_connection.close()
    
    return run_numbers, run_metric_value_lists, run_metric_time_lists


def get_most_recent_round_info(db_loc,
                               train_dataset_name,
                               model_architecture_name,
                               limited_mem,
                               arrival_pattern,
                               run_number,
                               training_loop,
                               al_method_name,
                               al_budget,
                               init_task_size,
                               unl_buffer_size):
    
    db_connection = _get_db_connection(db_loc)
    
    try:
        with db_connection:
            db_cursor = db_connection.cursor()
            db_cursor.execute("BEGIN ")
            
            # Before we open this transaction, let us first get the experiment id and the run id (if they exist). If either does not exist, raise
            # an error
            experiment_id   = get_experiment_id(db_cursor, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern)
            if experiment_id is None:   return None
            run_id          = get_run_id(db_cursor, experiment_id, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size)
            if run_id is None:          return None

            # Check the ALRound relation. If there are no entries, return None.
            db_cursor.execute("SELECT round_id, current_epoch, dataset_split_path, model_weight_path, opt_state_path, lr_state_path, completed_unix_time \
                              FROM ALRound \
                              WHERE experiment_id = ? AND run_id = ? \
                              ORDER BY round_id DESC \
                              LIMIT 1",
                              (experiment_id, run_id))
            most_recent_round_tuple = db_cursor.fetchone()
            
        if most_recent_round_tuple is None:
            db_connection.close()
            return None
        
        with db_connection:

            # Unpackage round information
            round_id, current_epoch, dataset_split_path, model_weight_path, opt_state_path, lr_sched_state_path, completed_unix_time = most_recent_round_tuple
            
            # Retrieve AL parameters
            db_cursor.execute("SELECT name, value \
                              FROM ALParam \
                              WHERE experiment_id = ? AND run_id = ? AND round_id = ?",
                              (experiment_id, run_id, round_id))
            al_params = db_cursor.fetchall()
            
            # Retrieve stop criteria
            db_cursor.execute("SELECT cname, pname, value \
                              FROM StopCriterion \
                              WHERE experiment_id = ? AND run_id = ? AND round_id = ?",
                              (experiment_id, run_id, round_id))
            stop_criteria = db_cursor.fetchall()  
            
            # Get augmentations
            db_cursor.execute("SELECT descriptor \
                              FROM Augment \
                              WHERE experiment_id = ? AND run_id = ? AND round_id = ? AND is_train = 1 \
                              ORDER BY aug_order ASC",
                              (experiment_id, run_id, round_id))
            train_augmentations = [x[0] for x in db_cursor.fetchall()]
            
            db_cursor.execute("SELECT descriptor \
                              FROM Augment \
                              WHERE experiment_id = ? AND run_id = ? AND round_id = ? AND is_train = 0 \
                              ORDER BY aug_order ASC",
                              (experiment_id, run_id, round_id))
            test_augmentations = [x[0] for x in db_cursor.fetchall()]

    except Exception as err:
        print("Error in retrieving most recent round info")
        raise err
    finally:    
        db_connection.close()
    
    # Return all the round info
    return  train_dataset_name, \
            model_architecture_name, \
            limited_mem, \
            arrival_pattern, \
            run_number, \
            training_loop, \
            al_method_name, \
            al_budget, \
            init_task_size, \
            unl_buffer_size, \
            round_id, \
            current_epoch, \
            dataset_split_path, \
            model_weight_path, \
            opt_state_path, \
            lr_sched_state_path, \
            completed_unix_time, \
            al_params, \
            train_augmentations, \
            test_augmentations, \
            stop_criteria


def get_all_experiments(db_loc):
    
    db_connection = _get_db_connection(db_loc)
    
    try:
        with db_connection:
            db_cursor = db_connection.cursor()
            db_cursor.execute("BEGIN ")
            db_cursor.execute("SELECT dataset_name, model_name, limited_mem, arrival_pattern FROM Experiment")
            experiments = db_cursor.fetchall()
    except Exception as err:
        print("Error in retrieving experiment groups")
        raise err
    finally:        
        db_connection.close()
    
    return experiments


def get_all_runs_from_experiment(db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern):
    
    db_connection = _get_db_connection(db_loc)
    
    try:
        with db_connection:
            db_cursor = db_connection.cursor()
            db_cursor.execute("BEGIN ")

            # Before we open this transaction, let us first get the experiment id and the run id (if they exist). If either does not exist, raise
            # an error
            experiment_id   = get_experiment_id(db_cursor, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern)
            if experiment_id is None:   raise ValueError("No matching experiment in DB")

            db_cursor.execute("SELECT run_number, training_loop, al_method, budget, init_task_size, unl_buffer_size FROM ALRun WHERE experiment_id = ?",
                              (experiment_id,))
            all_run_configs = db_cursor.fetchall()
    except Exception as err:
        print("Error in retrieving experiment")
        raise err
    finally:
        db_connection.close()
    
    return all_run_configs


def get_all_rounds_from_run(db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_num, training_loop, al_method, budget, init_task_size, unl_buffer_size):

    db_connection = _get_db_connection(db_loc)
    
    try:
        with db_connection:
            db_cursor = db_connection.cursor()
            db_cursor.execute("BEGIN ")
            
            # First, get the experiment id and the run id (if they exist). If either does not exist, raise
            # an error
            experiment_id   = get_experiment_id(db_cursor, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern)
            if experiment_id is None:   raise ValueError("No matching experiment in DB")
            run_id          = get_run_id(db_cursor, experiment_id, run_num, training_loop, al_method, budget, init_task_size, unl_buffer_size)
            if run_id is None:          raise ValueError("No matching run in DB")

            # Check the ALRound relation. If there are no entries, return None.
            db_cursor.execute("SELECT round_id, current_epoch, dataset_split_path, model_weight_path, opt_state_path, lr_state_path, completed_unix_time \
                              FROM ALRound \
                              WHERE experiment_id = ? AND run_id = ?\
                              ORDER BY round_id ASC",
                              (experiment_id, run_id))
            all_round_tuples = db_cursor.fetchall()

    except Exception as err:
        raise err
    finally:    
        db_connection.close()
    
    # Return all the round info
    return all_round_tuples


def get_al_rounds_without_valid_eval_dataset_metric(db_loc, train_dataset_name, metric_name, eval_dataset):

    db_connection = _get_db_connection(db_loc)
    
    try:
        with db_connection:
            db_cursor = db_connection.cursor()
            db_cursor.execute("BEGIN ")
            
            # Explanation:
            # 
            # We need to get all round information that have the specified training dataset, which is simply done by joining Experiment
            #   WHERE dataset_name = ?
            #
            # on ALRun and ALRound. Next, we can LEFT natural join the result on only those metric tuples that match the specified description
            #   WHERE name = ? AND eval_dataset = ?
            #
            # the result of which would have all AL rounds with (potentially) joined metric information.
            db_cursor.execute("SELECT dataset_name, model_name, limited_mem, arrival_pattern, run_number, training_loop, al_method, budget, init_task_size, unl_buffer_size,\
                                      round_id, current_epoch, dataset_split_path, model_weight_path, opt_state_path, lr_state_path, completed_unix_time, name, eval_dataset, value, computed_unix_time\
                              FROM ((SELECT * FROM Experiment WHERE dataset_name = ?)\
                                    NATURAL JOIN ALRun\
                                    NATURAL JOIN ALRound)\
                              LEFT NATURAL JOIN\
                                    (SELECT *\
                                    FROM Metric\
                                    WHERE name = ? AND eval_dataset = ?)\
                              WHERE ((completed_unix_time > computed_unix_time) OR (computed_unix_time IS NULL))", 
                              [train_dataset_name, metric_name, eval_dataset])

            al_rounds_without_valid_eval_dataset_metric = db_cursor.fetchall()

    except Exception as err:
        print("Error in retrieving AL rounds")
        raise err
    finally:    
        db_connection.close()

    return al_rounds_without_valid_eval_dataset_metric
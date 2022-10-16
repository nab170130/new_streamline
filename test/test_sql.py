import os
import streamline.persistence as persistence
import sqlite3
import unittest

class TestSQL(unittest.TestCase):
    
    def setUp(self):
        
        # Create a test database
        self.db_loc = "test_database.db"
        persistence.create_db(self.db_loc)
        
        self.db_connection = persistence._get_db_connection(self.db_loc)
        

    def test_create_db_relations(self):
         
        try:
            with self.db_connection:
                db_cursor = self.db_connection.cursor()
                
                # Make sure each relation is present
                db_cursor.execute("SELECT * FROM Experiment")
                db_cursor.execute("SELECT * FROM ALRun")
                db_cursor.execute("SELECT * FROM ALRound")
                db_cursor.execute("SELECT * FROM ALParam")
                db_cursor.execute("SELECT * FROM StopCriterion")
                db_cursor.execute("SELECT * FROM Metric")
                db_cursor.execute("SELECT * From Augment")
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
        

    def test_experiment_relation(self):
        
        try:
            with self.db_connection:
                db_cursor = self.db_connection.cursor()
                
                # Add an entry
                db_cursor.execute("INSERT INTO Experiment VALUES (0,'a','b',1,'c')")
                
                # Try adding the same entry again with the same primary key
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO Experiment VALUES (0,'a','b',1,'c')")

                # Try adding the same entry again with a different primary key but the same succeeding attributes
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO Experiment VALUES (1,'a','b',1,'c')")
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
            

    def test_alrun_relation(self):
        
        try:
            with self.db_connection:
                db_cursor = self.db_connection.cursor()
                
                # Try adding something directly to ALRun relation without any values for foreign key
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO ALRun VALUES (0,0,0,'a','b',1,1,1)")
                
                # Add an entry after satisfying foreign keys
                db_cursor.execute("INSERT INTO Experiment VALUES (0,'a','b',1,'c')")
                db_cursor.execute("INSERT INTO ALRun VALUES (0,0,0,'a','b',1,1,1)")
                
                # Try adding the same entry again with the same primary key
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO ALRun VALUES (0,0,0,'a','b',1,1,1)")
                
                # Try adding the same entry again with a different primary key but the same succeeding attributes
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO ALRun VALUES (1,0,0,'a','b',1,1,1)")
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
        

    def test_alround_relation(self):
        
        try:
            with self.db_connection:
                db_cursor = self.db_connection.cursor()
                
                # Try adding something directly to ALRound relation without any values for foreign key
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO ALRound VALUES (0,0,0,0,'a','b','c','d',0)")
                
                # Add an entry after satisfying foreign keys
                db_cursor.execute("INSERT INTO Experiment VALUES (0,'a','b',1,'c')")
                db_cursor.execute("INSERT INTO ALRun VALUES (0,0,0,'a','b',1,1,1)")
                db_cursor.execute("INSERT INTO ALRound VALUES (0,0,0,0,'a','b','c','d',0)")
                
                # Try adding the same entry again with the same primary key
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO ALRound VALUES (0,0,0,0,'a','b','c','d',0)")
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
        

    def test_alparam_relation(self):
        
        try:
            with self.db_connection:
                db_cursor = self.db_connection.cursor()
                
                # Try adding something directly to ALRound relation without any values for foreign key
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO ALParam VALUES (0,0,0,'param',4.0)")
                
                # Add an entry after satisfying pri
                db_cursor.execute("INSERT INTO Experiment VALUES (0,'a','b',1,'c')")
                db_cursor.execute("INSERT INTO ALRun VALUES (0,0,0,'a','b',1,1,1)")
                db_cursor.execute("INSERT INTO ALRound VALUES (0,0,0,0,'a','b','c','d',0)")
                db_cursor.execute("INSERT INTO ALParam VALUES (0,0,0,'param',4.0)")
                
                # Try adding the same entry again with the same primary key
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO ALParam VALUES (0,0,0,'param',4.0)")
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
        

    def test_stopcriterion_relation(self):
        
        try:
            with self.db_connection:
                db_cursor = self.db_connection.cursor()
                
                # Try adding something directly to ALRound relation without any values for foreign key
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO StopCriterion VALUES (0,0,0,'acc_crit','param',4.0)")
                
                # Add an entry after satisfying pri
                db_cursor.execute("INSERT INTO Experiment VALUES (0,'a','b',1,'c')")
                db_cursor.execute("INSERT INTO ALRun VALUES (0,0,0,'a','b',1,1,1)")
                db_cursor.execute("INSERT INTO ALRound VALUES (0,0,0,0,'a','b','c','d',0)")
                db_cursor.execute("INSERT INTO StopCriterion VALUES (0,0,0,'acc_crit','param',4.0)")
                
                # Try adding the same entry again with the same primary key
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO StopCriterion VALUES (0,0,0,'acc_crit','param',4.0)")
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
        

    def test_metric_relation(self):
        
        try:
            with self.db_connection:
                db_cursor = self.db_connection.cursor()
                
                # Try adding something directly to ALRound relation without any values for foreign key
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO Metric VALUES (0,0,0,'a','b',1.,1)")
                
                # Add an entry after satisfying pri
                db_cursor.execute("INSERT INTO Experiment VALUES (0,'a','b',1,'c')")
                db_cursor.execute("INSERT INTO ALRun VALUES (0,0,0,'a','b',1,1,1)")
                db_cursor.execute("INSERT INTO ALRound VALUES (0,0,0,0,'a','b','c','d',0)")
                db_cursor.execute("INSERT INTO Metric VALUES (0,0,0,'a','b',1.,1)")
                
                # Try adding the same entry again with the same primary key
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO Metric VALUES (0,0,0,'a','b',1.,1)")
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
            

    def test_augment_relation(self):
        
        try:
            with self.db_connection:
                db_cursor = self.db_connection.cursor()
                
                # Try adding something directly to ALRound relation without any values for foreign key
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO Augment VALUES (0,0,0,0,0,'randflip')")
                
                # Add an entry after satisfying pri
                db_cursor.execute("INSERT INTO Experiment VALUES (0,'a','b',1,'c')")
                db_cursor.execute("INSERT INTO ALRun VALUES (0,0,0,'a','b',1,1,1)")
                db_cursor.execute("INSERT INTO ALRound VALUES (0,0,0,0,'a','b','c','d',0)")
                db_cursor.execute("INSERT INTO Augment VALUES (0,0,0,0,0,'randflip')")
                
                # Try adding the same entry again with the same primary key
                with self.assertRaises(sqlite3.IntegrityError):
                    db_cursor.execute("INSERT INTO Augment VALUES (0,0,0,0,0,'randflip')")
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
        

    def test_cascading_deletes(self):
        
        try:
            with self.db_connection:
                db_cursor = self.db_connection.cursor()
                
                # Add entries to all tables
                db_cursor.execute("INSERT INTO Experiment VALUES (0,'a','b',1,'c')")
                db_cursor.execute("INSERT INTO ALRun VALUES (0,0,0,'a','b',1,1,1)")
                db_cursor.execute("INSERT INTO ALRound VALUES (0,0,0,0,'a','b','c','d',0)")
                db_cursor.execute("INSERT INTO ALParam VALUES (0,0,0,'param',4.0)")
                db_cursor.execute("INSERT INTO StopCriterion VALUES (0,0,0,'acc_crit','param',4.0)")
                db_cursor.execute("INSERT INTO Metric VALUES (0,0,0,'a','b',1.,1)")
                db_cursor.execute("INSERT INTO Augment VALUES (0,0,0,0,0,'randflip')")
                
                # Listed for posterity
                self.assertEqual(len(db_cursor.execute("SELECT * FROM ALParam").fetchall()), 1)
                self.assertEqual(len(db_cursor.execute("SELECT * FROM StopCriterion").fetchall()), 1)
                self.assertEqual(len(db_cursor.execute("SELECT * FROM Metric").fetchall()), 1)
                self.assertEqual(len(db_cursor.execute("SELECT * FROM Augment").fetchall()), 1)
                
                # Delete the entry from ALRun, which should cause cascading deletes in the relations listed below it in the above block
                db_cursor.execute("DELETE FROM ALRound")
                
                # Ensure that all other dependent relations are empty
                self.assertEqual(len(db_cursor.execute("SELECT * FROM ALParam").fetchall()), 0)
                self.assertEqual(len(db_cursor.execute("SELECT * FROM StopCriterion").fetchall()), 0)
                self.assertEqual(len(db_cursor.execute("SELECT * FROM Metric").fetchall()), 0)
                self.assertEqual(len(db_cursor.execute("SELECT * FROM Augment").fetchall()), 0)
                
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
        

    def test_add_round(self):
        
        # Declare some constants to use for testing
        train_dataset_name = "cifar10"
        model_architecture_name = "resnet18"
        limited_mem = 0
        arrival_pattern = "random"
        run_number = 0
        training_loop = "cross_entropy"
        al_method_name = "badge"
        al_budget = 50
        init_task_size = 50
        unl_buffer_size = 50
        round_number = 0
        current_epoch = 5
        dataset_split_path = "my_train_split.bin"
        model_weight_path = "my_weights.bin"
        optimizer_state_path = "my_sgd.bin"
        lr_schedule_state_path = "my_cosine_anneal.bin"
        completed_unix_time = 4
        al_params = [("embedding", 0)]
        train_augmentations = ["random_crop", "to_tensor"]
        test_augmentations = ["to_tensor"]
        stop_criteria = [("acc_crit", "acc_thr", 0.95)]
        metrics = [("acc", "cifar10_id", 0.6, 4)]
        
        # Add an active learning round
        persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                         round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                         al_params, train_augmentations, test_augmentations, stop_criteria, metrics)
        
        try:
            with self.db_connection:
                db_cursor = self.db_connection.cursor()
        
                # Check that there are entries in the Experiment and ALRun relation corresponding to the above parameters
                expected_experiment_tuple = (100, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern)
                expected_alrun_tuple = (100, 100, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size)
                actual_experiment_tuple = db_cursor.execute("SELECT * FROM Experiment").fetchone()
                actual_alrun_tuple = db_cursor.execute("SELECT * FROM ALRun").fetchone()
                self.assertTupleEqual(expected_experiment_tuple, actual_experiment_tuple)
                self.assertTupleEqual(expected_alrun_tuple, actual_alrun_tuple)
                
                # Check the other relations that have specific entries to populate
                expected_alround_tuple = (100, 100, round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time)
                expected_alparam_tuple = (100, 100, round_number, al_params[0][0], al_params[0][1])
                expected_stopcriterion_tuple = (100, 100, round_number, stop_criteria[0][0], stop_criteria[0][1], stop_criteria[0][2])
                expected_metric_tuple = (100, 100, round_number, metrics[0][0], metrics[0][1], metrics[0][2], metrics[0][3])
                expected_augmentation_tuples = set([(100, 100, round_number, 1, 0, train_augmentations[0]),
                                                    (100, 100, round_number, 1, 1, train_augmentations[1]),
                                                    (100, 100, round_number, 0, 0, test_augmentations[0])])
                
                actual_alround_tuple = db_cursor.execute("SELECT * FROM ALRound").fetchone()
                actual_alparam_tuple = db_cursor.execute("SELECT * FROM ALParam").fetchone()
                actual_stopcriterion_tuple = db_cursor.execute("SELECT * FROM StopCriterion").fetchone() 
                actual_metric_tuple = db_cursor.execute("SELECT * FROM Metric").fetchone()
                actual_augmentation_tuples = set(db_cursor.execute("SELECT * FROM Augment").fetchall())
                
                self.assertTupleEqual(expected_alround_tuple, actual_alround_tuple)
                self.assertTupleEqual(expected_alparam_tuple, actual_alparam_tuple)
                self.assertTupleEqual(expected_stopcriterion_tuple, actual_stopcriterion_tuple)
                self.assertTupleEqual(expected_metric_tuple, actual_metric_tuple)
                self.assertSetEqual(expected_augmentation_tuples, actual_augmentation_tuples)
                
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
        

    def test_add_round_staleness(self):
        
        # Declare some constants to use for testing
        train_dataset_name = "cifar10"
        model_architecture_name = "resnet18"
        limited_mem = 0
        arrival_pattern = "random"
        run_number = 0
        training_loop = "cross_entropy"
        al_method_name = "badge"
        al_budget = 50
        init_task_size = 50
        unl_buffer_size = 50
        round_number = 0
        current_epoch = 4
        dataset_split_path = "my_train_split.bin"
        model_weight_path = "my_weights.bin"
        optimizer_state_path = "my_sgd.bin"
        lr_schedule_state_path = "my_cosine_anneal.bin"
        completed_unix_time = 4
        al_params = [("embedding", 0)]
        train_augmentations = ["random_crop", "to_tensor"]
        test_augmentations = ["to_tensor"]
        stop_criteria = [("acc_crit", "acc_thr", 0.95)]
        metrics = [("acc", "cifar10_id", 0.6, 4)]
        
        # Add three active learning rounds in the order (0, 1, 0). There should be no entries corresponding to round 1 or more
        # due to staleness of the existing rounds.
        persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                         round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                         al_params, train_augmentations, test_augmentations, stop_criteria, metrics)
        persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                         round_number + 1, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                         al_params, train_augmentations, test_augmentations, stop_criteria, metrics)
        persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                         round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, "my_other_lr_sched.bin", completed_unix_time,
                         al_params, train_augmentations, test_augmentations, stop_criteria, metrics)
        
        try:
            with self.db_connection:
                db_cursor = self.db_connection.cursor()
        
                # Check that there are no entries corresponding to AL round 1 or greater in the ALRound relation and all its dependents.
                # Since there is only one AL run present, we do not need to additionally specify other parameters in the select that 
                # particularly target them.
                alround_tuples =        db_cursor.execute("SELECT * FROM ALRound WHERE round_id >= 1").fetchall()
                alparam_tuples =        db_cursor.execute("SELECT * FROM ALParam WHERE round_id >= 1").fetchall()
                stopcriterion_tuples =  db_cursor.execute("SELECT * FROM StopCriterion WHERE round_id >= 1").fetchall()
                metric_tuples =         db_cursor.execute("SELECT * FROM Metric WHERE round_id >= 1").fetchall()
                augment_tuples =        db_cursor.execute("SELECT * FROM Augment WHERE round_id >= 1").fetchall()
                
                self.assertEqual(len(alround_tuples), 0)
                self.assertEqual(len(alparam_tuples), 0)
                self.assertEqual(len(stopcriterion_tuples), 0)
                self.assertEqual(len(metric_tuples), 0)
                self.assertEqual(len(augment_tuples), 0)
        
                # Check that there are entries corresponding to round zero still present and are updated with the new lr_schedule name.
                alround_tuples =        db_cursor.execute("SELECT * FROM ALRound WHERE round_id = 0").fetchall()
                alparam_tuples =        db_cursor.execute("SELECT * FROM ALParam WHERE round_id = 0").fetchall()
                stopcriterion_tuples =  db_cursor.execute("SELECT * FROM StopCriterion WHERE round_id = 0").fetchall()
                metric_tuples =         db_cursor.execute("SELECT * FROM Metric WHERE round_id = 0").fetchall()
                augment_tuples =        db_cursor.execute("SELECT * FROM Augment WHERE round_id = 0").fetchall()
                
                self.assertEqual(len(alround_tuples), 1)
                self.assertEqual(len(alparam_tuples), 1)
                self.assertEqual(len(stopcriterion_tuples), 1)
                self.assertEqual(len(metric_tuples), 1)
                self.assertEqual(len(augment_tuples), 3)
        
                expected_alround_tuple = (100, 100, round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, "my_other_lr_sched.bin", completed_unix_time)
                actual_alround_tuple = db_cursor.execute("SELECT * FROM ALRound").fetchone()
                self.assertTupleEqual(expected_alround_tuple, actual_alround_tuple)
                
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
        

    def test_add_round_round_gap(self):
        
        # Declare some constants to use for testing
        train_dataset_name = "cifar10"
        model_architecture_name = "resnet18"
        limited_mem = 0
        arrival_pattern = "random"
        run_number = 0
        training_loop = "cross_entropy"
        al_method_name = "badge"
        al_budget = 50
        init_task_size = 50
        unl_buffer_size = 50
        round_number = 0
        current_epoch = 4
        dataset_split_path = "my_train_split.bin"
        model_weight_path = "my_weights.bin"
        optimizer_state_path = "my_sgd.bin"
        lr_schedule_state_path = "my_cosine_anneal.bin"
        completed_unix_time = 4
        al_params = [("embedding", 0)]
        train_augmentations = ["random_crop", "to_tensor"]
        test_augmentations = ["to_tensor"]
        stop_criteria = [("acc_crit", "acc_thr", 0.95)]
        metrics = [("acc", "cifar10_id", 0.6, 4)]
        
        # Add three active learning rounds in the order (1, 0, 2). The first attempt should fail since there is no round zero.
        # The third should fail because there is no round 1.
        with self.assertRaises(sqlite3.IntegrityError):
            persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                             round_number + 1, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                             al_params, train_augmentations, test_augmentations, stop_criteria, metrics)
        
        persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                         round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                         al_params, train_augmentations, test_augmentations, stop_criteria, metrics)
            
        with self.assertRaises(sqlite3.IntegrityError):
            persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                             round_number + 2, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                             al_params, train_augmentations, test_augmentations, stop_criteria, metrics)


    def test_add_metric_time_value_to_run(self):
            
        # Declare some constants to use for testing
        train_dataset_name = "cifar10"
        model_architecture_name = "resnet18"
        limited_mem = 0
        arrival_pattern = "random"
        run_number = 0
        training_loop = "cross_entropy"
        al_method_name = "badge"
        al_budget = 50
        init_task_size = 50
        unl_buffer_size = 50
        round_number = 0
        current_epoch = 4
        dataset_split_path = "my_train_split.bin"
        model_weight_path = "my_weights.bin"
        optimizer_state_path = "my_sgd.bin"
        lr_schedule_state_path = "my_cosine_anneal.bin"
        completed_unix_time = 4
        al_params = [("embedding", 0)]
        train_augmentations = ["random_crop", "to_tensor"]
        test_augmentations = ["to_tensor"]
        stop_criteria = [("acc_crit", "acc_thr", 0.95)]
        metrics = [("acc", "cifar10_id", 0.6, 4)]
            
        # Add a round
        persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                         round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                         al_params, train_augmentations, test_augmentations, stop_criteria, metrics)
        
        # Create a new metric list
        new_metric = "mce"
        eval_dataset = "cifar10-c"
        new_metric_value_in_run = 0.4
        new_metric_time_in_run = 2
        
        try:
            db_cursor = self.db_connection.cursor()
            
            # Add the new list
            persistence.add_metric_time_value_to_run(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                                                     round_number, new_metric, eval_dataset, new_metric_value_in_run, new_metric_time_in_run)
                
            # Try updating one of the accuracy metric values
            new_acc_metric = "acc"
            acc_eval_dataset = "cifar10_id"
            new_acc_metric_value_in_run = 0.7
            new_acc_metric_time_in_run = 4
            
            # Update the new list
            persistence.add_metric_time_value_to_run(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                                                     round_number, new_acc_metric, acc_eval_dataset, new_acc_metric_value_in_run, new_acc_metric_time_in_run)
            
            # Ensure that the new metric value was added. Since there is only 1 AL run, we only need to specify the metric name.
            metric_values = db_cursor.execute("SELECT value, computed_unix_time FROM Metric WHERE name = ?", [new_metric]).fetchall()
            self.assertAlmostEqual(new_metric_value_in_run, metric_values[0][0])
            self.assertEqual(new_metric_time_in_run, metric_values[0][1])
            
            # Ensure that the new metric values were updated. Since there is only 1 AL run, we only need to specify the metric name.
            metric_values = db_cursor.execute("SELECT value, computed_unix_time FROM Metric WHERE name = ?", [new_acc_metric]).fetchall()
            self.assertAlmostEqual(new_acc_metric_value_in_run, metric_values[0][0])
            self.assertEqual(new_acc_metric_time_in_run, metric_values[0][1])
            
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
    

    def test_get_metric_values_computed_times_over_all_runs(self):
        
        # Declare some constants to use for testing. This time, a bunch of different metrics will exist for multiple runs
        train_dataset_name = "cifar10"
        model_architecture_name = "resnet18"
        limited_mem = 0
        arrival_pattern = "random"
        training_loop = "cross_entropy"
        al_method_name = "badge"
        al_budget = 50
        init_task_size = 50
        unl_buffer_size = 50
        current_epoch = -1
        dataset_split_path = "my_train_split.bin"
        model_weight_path = "my_weights.bin"
        optimizer_state_path = "my_sgd.bin"
        lr_schedule_state_path = "my_cosine_anneal.bin"
        completed_unix_time = 3
        al_params = [("embedding", 0)]
        train_augmentations = ["random_crop", "to_tensor"]
        test_augmentations = ["to_tensor"]
        stop_criteria = [("acc_crit", "acc_thr", 0.95)]
        metrics = [("acc", "cifar10_id", 0.6, 4)]
        
        n_runs = 2
        n_rounds = 3
        
        # Add some rounds to two runs
        for run_num in range(n_runs):
            for round_num in range(n_rounds):
                metrics = [("acc", "cifar10_id", 0.1 * run_num + 0.2 * round_num, 4),
                           ("mce", "cifar10_c", 0.1 * run_num + 0.5 / (1 + round_num), 4)]
                persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_num, training_loop, al_method_name, 
                                         al_budget, init_task_size, unl_buffer_size, round_num, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, 
                                         lr_schedule_state_path, completed_unix_time, al_params, train_augmentations, test_augmentations, stop_criteria, metrics)
    
        # Get metric values corresponding to accuracy (acc) and mce 
        acc_run_nums, acc_vals, acc_times = persistence.get_valid_metric_values_computed_times_over_all_runs(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, training_loop, al_method_name, 
                                                                                                             al_budget, init_task_size, unl_buffer_size, "acc", "cifar10_id")
        mce_run_nums, mce_vals, mce_times = persistence.get_valid_metric_values_computed_times_over_all_runs(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, training_loop, al_method_name, 
                                                                                                             al_budget, init_task_size, unl_buffer_size, "mce", "cifar10_c")
        
        # Ensure that there are two runs, 3 rounds per run for each list
        self.assertEqual(len(acc_vals), n_runs)
        for round_list in acc_vals:
            self.assertEqual(len(round_list), n_rounds)
            
        self.assertEqual(len(mce_vals), n_runs)
        for round_list in mce_vals:
            self.assertEqual(len(round_list), n_rounds)

        self.assertEqual(len(acc_times), n_runs)
        for round_list in acc_times:
            self.assertEqual(len(round_list), n_rounds)
            
        self.assertEqual(len(mce_times), n_runs)
        for round_list in mce_times:
            self.assertEqual(len(round_list), n_rounds)

        # Check that the run numbers are correct
        self.assertListEqual(acc_run_nums, list(range(n_runs)))
        self.assertListEqual(mce_run_nums, list(range(n_runs)))
        
        # Check that there are only the accuracy metric values by comparing against the previous formula
        for run_num in range(n_runs):
            for round_num in range(n_rounds):
                expected_value = 0.1 * run_num + 0.2 * round_num
                expected_time = 4
                self.assertEqual(acc_vals[run_num][round_num], expected_value)
                self.assertEqual(acc_times[run_num][round_num], expected_time)
                
        # Check that there are only the mce metric values by comparing against the previous formula
        for run_num in range(n_runs):
            for round_num in range(n_rounds):
                expected_value = 0.1 * run_num + 0.5 / (1 + round_num)
                self.assertEqual(mce_vals[run_num][round_num], expected_value)
                self.assertEqual(acc_times[run_num][round_num], expected_time)

        # Now, insert a couple "invalid" metrics
        # Metric is computed before round finishes
        completed_unix_time = 5                     
        metrics = [("acc", "cifar10_id", .9, 4)]
        persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, 0, training_loop, al_method_name, 
                                 al_budget, init_task_size, unl_buffer_size, n_rounds, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, 
                                 lr_schedule_state_path, completed_unix_time, al_params, train_augmentations, test_augmentations, stop_criteria, metrics)
        
        # Metric's computed time is NULL
        completed_unix_time = 4
        metrics = [("acc", "cifar10_id", .9, None)]
        persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, 0, training_loop, al_method_name, 
                                 al_budget, init_task_size, unl_buffer_size, n_rounds + 1, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, 
                                 lr_schedule_state_path, completed_unix_time, al_params, train_augmentations, test_augmentations, stop_criteria, metrics)
        
        # Metric is not finished training
        current_epoch = 5
        metrics = [("acc", "cifar10_id", .9, 5)]
        persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, 0, training_loop, al_method_name, 
                                 al_budget, init_task_size, unl_buffer_size, n_rounds + 2, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, 
                                 lr_schedule_state_path, completed_unix_time, al_params, train_augmentations, test_augmentations, stop_criteria, metrics)

        # Get the accuracy runs back
        acc_run_nums, acc_vals, acc_times = persistence.get_valid_metric_values_computed_times_over_all_runs(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, training_loop, al_method_name, 
                                                                                                             al_budget, init_task_size, unl_buffer_size, "acc", "cifar10_id")

        # Make sure that the zeroth round does not have the last 3 rounds inserted above.
        for idx, acc_run_num in enumerate(acc_run_nums):
            if acc_run_num == 0:
                self.assertEqual(len(acc_vals[idx]), n_rounds)


    def test_get_most_recent_round_info(self):
        
        # Declare some constants to use for testing.
        train_dataset_name = "cifar10"
        model_architecture_name = "resnet18"
        limited_mem = 0
        arrival_pattern = "random"
        training_loop = "cross_entropy"
        al_method_name = "badge"
        run_number = 0
        al_budget = 50
        init_task_size = 50
        unl_buffer_size = 50
        round_number = 0
        current_epoch = 4
        dataset_split_path = "my_train_split.bin"
        model_weight_path = "my_weights.bin"
        optimizer_state_path = "my_sgd.bin"
        lr_schedule_state_path = "my_cosine_anneal.bin"
        completed_unix_time = 4
        al_params = [("embedding", 0)]
        train_augmentations = ["random_crop", "to_tensor"]
        test_augmentations = ["to_tensor"]
        stop_criteria = [("acc_crit", "acc_thr", 0.95)]
        metrics = [("acc", "cifar10_id", 0.6, 4)]
        
        # Add two identical rounds
        persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                         round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                         al_params, train_augmentations, test_augmentations, stop_criteria, metrics)
        persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                         round_number + 1, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                         al_params, train_augmentations, test_augmentations, stop_criteria, metrics)

        # Get the most recent round info, which should be the tuple with round number as round_number + 1        
        actual_tuple = persistence.get_most_recent_round_info(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size)
        expected_tuple = (train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size, round_number + 1, current_epoch, dataset_split_path, 
                          model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time, al_params, train_augmentations, test_augmentations, stop_criteria)
        self.assertTupleEqual(expected_tuple, actual_tuple)
        

    def test_get_all_experiments(self):
        
        # Add some experiments to the database
        fab_ids = [0,1,2]
        datasets = ["cifar10", "cifar100", "svhn"]
        models = ["resnet18", "resnet18", "lenet"]
        limited_mem = [0, 0, 0]
        arrival_pattern = ["random", "random", "random"]
        expected_tuples = list(zip(datasets, models, limited_mem, arrival_pattern))
        
        try:
            # Add and commit values to the database
            with self.db_connection:
                db_cursor = self.db_connection.cursor()
                db_cursor.executemany("INSERT INTO Experiment VALUES (?,?,?,?,?)", zip(fab_ids, datasets, models, limited_mem, arrival_pattern))
                   
            # Get the list of experiments and check that they align with what was added
            actual_tuples = persistence.get_all_experiments(self.db_loc)
            self.assertSetEqual(set(expected_tuples), set(actual_tuples))
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
            

    def test_get_all_run_configs_from_experiment(self):
        
        # Add some runs to the database
        train_dataset_name = "cifar10"
        model_architecture_name = "resnet18"
        limited_mem = 0
        arrival_pattern = "random"
        fab_ids = [0,1,2]
        training_loops = ["cross_entropy", "cross_entropy", "cross_entropy"]
        al_methods = ["badge", "badge", "ours"]
        run_nums = [0, 1, 2]
        budgets = [50, 50, 50]
        init_task_sizes = [50, 50, 50]
        unl_buffer_sizes = [50, 50, 50]
        expected_tuples = zip(fab_ids, run_nums, training_loops, al_methods, budgets, init_task_sizes, unl_buffer_sizes)
        
        try:
            # Add and commit values to the database
            with self.db_connection:
                db_cursor = self.db_connection.cursor()
                db_cursor.execute("INSERT INTO Experiment VALUES (?,?,?,?,?)", (0, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern))       
                for fab_id, training_loop, al_method, run_num, budget, init_task_size, unl_buffer_size in expected_tuples:
                    db_cursor.execute("INSERT INTO ALRun VALUES (?,?,?,?,?,?,?,?)", (0, fab_id, training_loop, al_method, run_num, budget, init_task_size, unl_buffer_size))
                   
            # Get the list of run configurations and see if they match what was supposed to be added
            actual_tuples = persistence.get_all_runs_from_experiment(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern)
            for expected_tuple, actual_tuple in zip(expected_tuples, actual_tuples):
                self.assertEqual(expected_tuple, actual_tuple)
        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")
        

    def test_get_all_rounds_from_run(self):

        # Declare some constants to use for testing
        train_dataset_name = "cifar10"
        model_architecture_name = "resnet18"
        limited_mem = 0
        arrival_pattern = "random"
        run_number = 0
        training_loop = "cross_entropy"
        al_method_name = "badge"
        al_budget = 50
        init_task_size = 50
        unl_buffer_size = 50
        round_number = 0
        current_epoch = 4
        dataset_split_path = "my_train_split.bin"
        model_weight_path = "my_weights.bin"
        optimizer_state_path = "my_sgd.bin"
        lr_schedule_state_path = "my_cosine_anneal.bin"
        completed_unix_time = 4
        expected_tuples = [(round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, 
                            lr_schedule_state_path, completed_unix_time),
                            (round_number + 1, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, 
                            lr_schedule_state_path, completed_unix_time + 1)]

        try:
            
            # Insert two rounds
            persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                         round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                         [], [], [], [], [])
            persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                         round_number + 1, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time + 1,
                         [], [], [], [], [])

            # Get the list of rounds and see if they match what was supposed to be added
            actual_tuples = persistence.get_all_rounds_from_run(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, 
                                                                al_budget, init_task_size, unl_buffer_size)
            for expected_tuple, actual_tuple in zip(expected_tuples, actual_tuples):
                self.assertEqual(expected_tuple, actual_tuple)

        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")


    def test_get_al_rounds_without_valid_eval_dataset_metric(self):

        # Declare some constants to use for testing.
        train_dataset_name = "cifar10"
        model_architecture_name = "resnet18"
        limited_mem = 0
        arrival_pattern = "random"
        run_number = 0
        training_loop = "cross_entropy"
        al_method_name = "badge"
        al_budget = 50
        init_task_size = 50
        unl_buffer_size = 50
        round_number = 0
        current_epoch = 4
        dataset_split_path = "my_train_split.bin"
        model_weight_path = "my_weights.bin"
        optimizer_state_path = "my_sgd.bin"
        lr_schedule_state_path = "my_cosine_anneal.bin"
        completed_unix_time = 4
        al_params = [("embedding", 0)]
        train_augmentations = ["random_crop", "to_tensor"]
        test_augmentations = ["to_tensor"]
        stop_criteria = [("acc_crit", "acc_thr", 0.95)]
        metrics = [("acc", "cifar10_id", 0.6, 5)]

        try:
            with self.db_connection:
                db_cursor = self.db_connection.cursor()

                # Add three rounds, where one does not have acc metric, one has an outdated metric, and one is fine.
                persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                                         round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                                         al_params, train_augmentations, test_augmentations, stop_criteria, metrics)                            # Fine
                persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                                         round_number + 1, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                                         al_params, train_augmentations, test_augmentations, stop_criteria, [("acc", "cifar10_id", 0.7, 3)])    # Outdated
                persistence.add_al_round(self.db_loc, train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                                         round_number + 2, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                                         al_params, train_augmentations, test_augmentations, stop_criteria, [("mce", "cifar10_id", 0.7,5)])     # DNE

                # Get the tuples returned 
                actual_invalid_tuples = persistence.get_al_rounds_without_valid_eval_dataset_metric(self.db_loc, train_dataset_name, "acc", "cifar10_id")
                expected_invalid_tuples = [(train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size, round_number + 1, current_epoch, dataset_split_path, 
                                                model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time, "acc", "cifar10_id", 0.7, 3),
                                           (train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size, round_number + 2, current_epoch, dataset_split_path, 
                                                model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time, None, None, None, None)]

                # See if the returned tuples align
                self.assertSetEqual(set(expected_invalid_tuples), set(actual_invalid_tuples))

                # Try with mce
                actual_invalid_tuples = persistence.get_al_rounds_without_valid_eval_dataset_metric(self.db_loc, train_dataset_name, "mce", "cifar10_id")
                expected_invalid_tuples = [(train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size, round_number, current_epoch, dataset_split_path, 
                                                model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time, None, None, None, None),
                                           (train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size, round_number + 1, current_epoch, dataset_split_path, 
                                                model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time, None, None, None, None)]

                # See if the returned tuples align
                self.assertSetEqual(set(expected_invalid_tuples), set(actual_invalid_tuples))

                # Add a round with an invalid metric but on a different set.
                persistence.add_al_round(self.db_loc, "imagenet", model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size,
                                         round_number, current_epoch, dataset_split_path, model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time,
                                         al_params, train_augmentations, test_augmentations, stop_criteria, [("acc", "cifar10_id", 0.7, 3)])    # Outdated

                # Get the tuples returned. The newly added tuple should not affect anything.
                actual_invalid_tuples = persistence.get_al_rounds_without_valid_eval_dataset_metric(self.db_loc, train_dataset_name, "acc", "cifar10_id")
                expected_invalid_tuples = [(train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size, round_number + 1, current_epoch, dataset_split_path, 
                                                model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time, "acc", "cifar10_id", 0.7, 3),
                                           (train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, run_number, training_loop, al_method_name, al_budget, init_task_size, unl_buffer_size, round_number + 2, current_epoch, dataset_split_path, 
                                                model_weight_path, optimizer_state_path, lr_schedule_state_path, completed_unix_time, None, None, None, None)]

                # See if the returned tuples align
                self.assertSetEqual(set(expected_invalid_tuples), set(actual_invalid_tuples))

        except Exception as err:
            self.fail(F"Ran into exception during test: {err}")


    def tearDown(self):
        
        # Close the open connection
        self.db_connection.close()
        
        # Delete the test database
        os.remove(self.db_loc)
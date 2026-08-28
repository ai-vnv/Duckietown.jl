# FJ8.4b — six-solver comparison

Frozen protocol: `configs/planning/fj8_evaluation.yaml`. 20 evaluation seeds (disjoint from the development seeds used for calibration), horizon 150, one shared evaluator, planner RNG 2026.

**There is no combined ranking.** Task performance and computational cost are separate blocks and are not comparable to each other.

## Task performance

```
solver         return     median                95% CI   progress   mean|d|    max|d|  mean|phi|    speed     len
-----------------------------------------------------------------------------------------------------------------
q_learning      11.63      12.86         [-13.9, 27.5]      20.08    0.0342      0.25     0.1494   0.1391   145.6
sarsa           11.92      12.86         [-13.0, 27.5]      20.07    0.0323      0.25     0.1436   0.1408   144.8
sac             -1.32       4.75           [-9.2, 5.0]      14.52    0.0596    0.1394     0.1171   0.0979   150.0
td3           -213.78    -216.08      [-222.4, -205.0]       4.48     0.033    0.0723     0.1241   0.0303   150.0
mcts@1k          6.78       7.75            [4.6, 8.8]      15.96    0.0859      0.25     0.2522   0.1125   150.0
dpw@1k        -214.46    -211.59      [-228.2, -197.7]       7.13    0.0744      0.25     0.3558   0.0889    85.6
```

## Safety, stops and ducks

`compliance` = (stop encounters − violations) / stop encounters, where an encounter is a `passed_stop` event. `n/a` means no stop sign was ever reached, which is not the same as perfect compliance.

```
solver        eps   offroad   collide   duck hit   stop enc   full stop   violation   compliance   duck act   crossings  reasons
--------------------------------------------------------------------------------------------------------------
q_learning     20         1         0          0          8           8           0       100.0%        140          20  Dict("offroad" => 1, "in_progress" => 19)
sarsa          20         1         0          0          8           8           0       100.0%        140          20  Dict("offroad" => 1, "in_progress" => 19)
sac            20         0         0          0         20          18           2        90.0%        140          20  Dict("in_progress" => 20)
td3            20         0         0          0          0          20           0          n/a          0           0  Dict("in_progress" => 20)
mcts@1k        20         0         0          0          4           5           0       100.0%        134          20  Dict("in_progress" => 20)
dpw@1k         20        13         6          0          7           4           7         0.0%          7           1  Dict("offroad" => 13, "other_collision" => 6, "in_progress" => 1)
```

## Computational cost

`gen/act = 0` is measured and means the policy performs no generative planning; `n/a` would mean unmeasured. Learned policies do tabular or network inference, planners run hundreds to thousands of model simulations — the latency columns are not like-for-like and must not be read as a single efficiency axis.

```
solver                      family    ms mean    ms p50    ms p95    gen/act    iters   act nodes   state nodes
------------------------------------------------------------------------------------------------------
q_learning       tabular (learned)      0.036     0.021     0.061        0.0        -           -             -
sarsa            tabular (learned)      0.035      0.02     0.033        0.0        -           -             -
sac                 deep (learned)      0.059      0.04     0.054        0.0        -           -             -
td3                 deep (learned)      0.052     0.036     0.051        0.0        -           -             -
mcts@1k     discrete online planning    125.464   128.458   169.266      999.4     36.0       259.0          37.0
dpw@1k      continuous online planning    106.847    104.39    213.62      721.7     35.0        34.9          35.9
```

## Paired per-seed differences in return

Every solver ran the same seeds, so these are differences on identical initial conditions rather than a comparison of independent group means. With n = 20 these are descriptive; no superiority claim is made.

```
pair                      metric   mean diff     median                  95% CI    a>b    b>a
------------------------------------------------------------------------------------------
mcts@1k - q_learning         ret       -4.85     -9.504         [-21.08, 20.86]      1     19
mcts@1k - sarsa              ret      -5.142     -9.504         [-21.09, 19.98]      1     19
mcts@1k - sac                ret       8.094      2.854            [1.26, 16.9]     14      6
mcts@1k - td3                ret     220.559    221.419        [211.03, 229.97]     20      0
dpw@1k - q_learning          ret     -226.09   -241.939      [-249.65, -196.16]      0     20
dpw@1k - sarsa               ret    -226.382   -241.939      [-249.66, -197.03]      0     20
dpw@1k - sac                 ret    -213.146   -212.588      [-227.83, -195.99]      0     20
dpw@1k - td3                 ret      -0.681      -8.39         [-15.95, 17.53]      9     11
mcts@1k - dpw@1k             ret     221.239    218.459        [204.38, 234.94]     20      0
```

## Planning cost by position in the episode

FJ8.4a measured a 3.8x spread in generative cost across states. This is where a high p95 comes from.

### mcts@1k

```
  episode fraction        n    ms mean     ms p95    gen/act
----------------------------------------------------------
           0.0-0.2      600     130.44     169.27     1019.2
           0.2-0.4      600     130.24     174.13     1020.9
           0.4-0.6      600     129.26     172.42     1024.8
           0.6-0.8      600     113.74     162.92      913.9
           0.8-1.0      600     123.64     165.62     1018.3
```

### dpw@1k

```
  episode fraction        n    ms mean     ms p95    gen/act
----------------------------------------------------------
           0.0-0.2      350     155.02     230.09     1019.1
           0.2-0.4      342     122.43     216.78      835.3
           0.4-0.6      343      136.6     221.61      940.3
           0.6-0.8      342      88.82     164.58      602.0
           0.8-1.0      336      28.79      77.15      195.1
```

## Planner budget sensitivity (planner-only)

Not additional rows of the six-solver table: a separate study on 5 seeds at horizon 60.

```
solver        return     median                95% CI   progress   mean|d|    max|d|  mean|phi|    speed     len
----------------------------------------------------------------------------------------------------------------
mcts@500         7.0        4.7           [3.6, 12.9]       6.92    0.0653      0.25     0.1771   0.1187    60.0
dpw@500      -165.27    -199.22       [-210.1, -86.1]       3.93     0.079    0.2462     0.4078   0.0918    48.0
mcts@1000       5.87       3.24           [2.9, 11.0]       6.22    0.0405    0.1288     0.2298   0.1081    60.0
dpw@1000      -30.15     -41.39        [-43.9, -16.4]       5.73     0.076      0.25     0.2442    0.101    60.0
mcts@2000       6.73       4.57           [3.6, 12.0]       6.26    0.0457    0.1311     0.1725   0.1074    60.0
dpw@2000       -26.9     -39.69        [-43.1, -10.7]       5.33    0.0545    0.1983     0.2123   0.0918    60.0
```

```
solver                     family    ms mean    ms p50    ms p95    gen/act    iters   act nodes   state nodes
-----------------------------------------------------------------------------------------------------
mcts@500   discrete online planning     60.924    57.026    93.158      489.9     18.0       133.0          19.0
dpw@500    continuous online planning     41.975    36.027     102.3      292.8     18.0        18.0          19.0
mcts@1000  discrete online planning    130.568   136.917   166.888     1047.2     36.0       259.0          37.0
dpw@1000   continuous online planning      127.3   129.428   226.426      886.2     35.0        35.0          36.0
mcts@2000  discrete online planning    255.257   266.994   328.684     2030.0     71.0       504.0          72.0
dpw@2000   continuous online planning    265.121   254.215   440.188     1834.8     70.0        70.0          71.0
```

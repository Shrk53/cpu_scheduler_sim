/*
 * proj2.c  --  Lab 05: CPU Scheduling Simulator
 *
 * Simulates three CPU scheduling policies in a time-driven manner:
 *     1) FCFS  - First Come First Served   (non-preemptive)
 *     2) RR    - Round Robin               (preemptive, fixed time quantum)
 *     3) SJF   - Shortest Job First        (non-preemptive)
 *
 * Build:   gcc -Wall -Wextra -o proj2 proj2.c
 * Usage:   proj2 input_file [FCFS|RR|SJF] [time_quantum]
 *            proj2 input.1 FCFS
 *            proj2 input.1 RR 2      (quantum is required for RR)
 *            proj2 input.1 SJF
 *
 * Input file format:
 *     <number of tasks>
 *     <pid> <arrival_time> <burst_time>      (one line per task, all integers)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>   /* strcasecmp */

/* ------------------------------------------------------------------ */
/* Constants and data structures                                       */
/* ------------------------------------------------------------------ */

#define MAX_TASKS     20      /* max tasks the simulator accepts            */
#define MAX_QUEUE     20      /* max ready-queue length (per assignment)    */
#define MAX_SEGMENTS  10000   /* max recorded CPU bursts (for RR trace)     */

typedef enum { ALGO_FCFS, ALGO_RR, ALGO_SJF } Algo;

/* One task (process) read from the input file. */
typedef struct {
    int pid;          /* unique process id                                  */
    int arrival;      /* arrival time (ms)                                  */
    int burst;        /* total CPU time requested (ms)                      */
    int remaining;    /* CPU time still needed                              */
    int start;        /* first time the task got the CPU (-1 = not yet)     */
    int end;          /* time the task completed                            */
} Task;

/* One continuous stretch of CPU time given to a task (used by the RR table). */
typedef struct {
    int pid;
    int start;
    int end;
} Segment;

/* All tasks, stored in the order they appear in the input file. */
static Task tasks[MAX_TASKS];
static int  num_tasks = 0;

/* Static-array ready queue.  It holds INDICES into tasks[].
 * ready_q[0] is the front of the queue; ready_count is the current length. */
static int ready_q[MAX_QUEUE];
static int ready_count = 0;

/* Execution trace, plus the order in which tasks completed. */
static Segment segments[MAX_SEGMENTS];
static int     num_segments = 0;
static int     finish_order[MAX_TASKS];
static int     num_finished = 0;

/* ------------------------------------------------------------------ */
/* Ready-queue helpers (static array implementation)                   */
/* ------------------------------------------------------------------ */

/* Append a task index to the back of the ready queue. */
static void rq_push(int idx)
{
    if (ready_count >= MAX_QUEUE) {
        fprintf(stderr, "Error: ready queue overflow (max %d)\n", MAX_QUEUE);
        exit(1);
    }
    ready_q[ready_count++] = idx;
}

/* Remove and return the element at position pos (0 = front).
 * Elements behind it shift forward so the queue stays contiguous. */
static int rq_remove_at(int pos)
{
    int idx = ready_q[pos];
    for (int i = pos; i < ready_count - 1; i++)
        ready_q[i] = ready_q[i + 1];
    ready_count--;
    return idx;
}

/* Print the ready queue contents as  [P1 P2 P3]  (or [ ] when empty). */
static void rq_print(void)
{
    printf("[");
    if (ready_count == 0) printf(" ");
    for (int i = 0; i < ready_count; i++)
        printf("%sP%d", i ? " " : "", tasks[ready_q[i]].pid);
    printf("]");
}

/* ------------------------------------------------------------------ */
/* Scheduling algorithms                                               */
/* Each one removes a task from the ready queue and returns its index  */
/* in tasks[], or -1 if the ready queue is empty (CPU idle).           */
/* ------------------------------------------------------------------ */

/* FCFS: the task that has been waiting longest (queue front) runs next. */
static int pick_fcfs(void)
{
    if (ready_count == 0) return -1;
    return rq_remove_at(0);
}

/* RR: also takes the queue front.  What makes it "round robin" is that the
 * simulator preempts the task after one quantum and puts it at the back. */
static int pick_rr(void)
{
    if (ready_count == 0) return -1;
    return rq_remove_at(0);
}

/* SJF: choose the ready task with the smallest burst time.
 * Ties: earlier arrival first, then earlier position in the input file. */
static int pick_sjf(void)
{
    if (ready_count == 0) return -1;

    int best = 0;                                /* position within queue */
    for (int i = 1; i < ready_count; i++) {
        const Task *a = &tasks[ready_q[i]];
        const Task *b = &tasks[ready_q[best]];
        if (a->burst < b->burst ||
            (a->burst == b->burst && a->arrival < b->arrival) ||
            (a->burst == b->burst && a->arrival == b->arrival &&
             ready_q[i] < ready_q[best]))
            best = i;
    }
    return rq_remove_at(best);
}

typedef int (*SchedFn)(void);

/* ------------------------------------------------------------------ */
/* Input handling                                                      */
/* ------------------------------------------------------------------ */

/* Read tasks from the file.  Returns 0 on success, -1 on error. */
static int read_tasks(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open input file '%s'\n", filename);
        return -1;
    }

    int n;
    if (fscanf(fp, "%d", &n) != 1 || n <= 0) {
        fprintf(stderr, "Error: first line must be a positive task count\n");
        fclose(fp);
        return -1;
    }
    if (n > MAX_TASKS) {
        fprintf(stderr, "Error: at most %d tasks are supported (file has %d)\n",
                MAX_TASKS, n);
        fclose(fp);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        Task *t = &tasks[i];
        if (fscanf(fp, "%d %d %d", &t->pid, &t->arrival, &t->burst) != 3) {
            fprintf(stderr, "Error: could not read task %d of %d\n", i + 1, n);
            fclose(fp);
            return -1;
        }
        if (t->arrival < 0 || t->burst <= 0) {
            fprintf(stderr, "Error: task %d has invalid arrival/burst time\n",
                    t->pid);
            fclose(fp);
            return -1;
        }
        for (int j = 0; j < i; j++) {
            if (tasks[j].pid == t->pid) {
                fprintf(stderr, "Error: duplicate pid %d\n", t->pid);
                fclose(fp);
                return -1;
            }
        }
    }
    fclose(fp);
    num_tasks = n;
    return 0;
}

/* ------------------------------------------------------------------ */
/* The simulator                                                       */
/* ------------------------------------------------------------------ */

static void simulate(Algo algo, int quantum)
{
    SchedFn pick = (algo == ALGO_FCFS) ? pick_fcfs :
                   (algo == ALGO_RR)   ? pick_rr   : pick_sjf;

    /* reset per-task state and global bookkeeping */
    for (int i = 0; i < num_tasks; i++) {
        tasks[i].remaining = tasks[i].burst;
        tasks[i].start     = -1;
        tasks[i].end       = -1;
    }
    ready_count = num_segments = num_finished = 0;

    int time      = 0;    /* current time slot [time, time+1)               */
    int running   = -1;   /* index of task holding the CPU, -1 = none       */
    int slice     = 0;    /* ms the running task has used in this turn (RR) */
    int seg_start = 0;    /* when the running task's current turn began     */
    int preempted = -1;   /* task preempted at the end of the last slot     */

    while (num_finished < num_tasks) {

        /* 1. Admit tasks arriving at this instant into the ready queue.
         *    (Scanning in file order keeps simultaneous arrivals stable.) */
        for (int i = 0; i < num_tasks; i++) {
            if (tasks[i].arrival == time) {
                rq_push(i);
                printf("        t=%-3d  P%d arrives (burst %d)\n",
                       time, tasks[i].pid, tasks[i].burst);
            }
        }

        /* 2. A task preempted at the end of the previous slot goes to the
         *    back of the queue, behind any task that just arrived. */
        if (preempted >= 0) {
            rq_push(preempted);
            preempted = -1;
        }

        /* 3. If the CPU is free, call the scheduling algorithm. */
        if (running < 0) {
            running = pick();
            if (running >= 0) {
                slice     = 0;
                seg_start = time;
                if (tasks[running].start < 0) {
                    tasks[running].start = time;
                    printf("        t=%-3d  P%d starts running\n",
                           time, tasks[running].pid);
                } else {
                    printf("        t=%-3d  P%d resumes running\n",
                           time, tasks[running].pid);
                }
            }
        }

        /* 4. Execute one time unit (or sit idle) and report it. */
        if (running < 0) {
            printf("[%3d-%3d] idle               ready=", time, time + 1);
            rq_print();
            printf("\n");
        } else {
            Task *t = &tasks[running];
            t->remaining--;
            slice++;

            printf("[%3d-%3d] run P%-3d rem=%-3d   ready=",
                   time, time + 1, t->pid, t->remaining);
            rq_print();
            printf("\n");

            if (t->remaining == 0) {
                /* task finished */
                t->end = time + 1;
                segments[num_segments++] =
                    (Segment){ t->pid, seg_start, time + 1 };
                finish_order[num_finished++] = running;
                printf("        t=%-3d  P%d completes\n", time + 1, t->pid);
                running = -1;
            } else if (algo == ALGO_RR && slice == quantum) {
                /* quantum used up: preempt; requeued at start of next slot */
                segments[num_segments++] =
                    (Segment){ t->pid, seg_start, time + 1 };
                printf("        t=%-3d  P%d quantum expired, preempted\n",
                       time + 1, t->pid);
                preempted = running;
                running   = -1;
            }
        }
        time++;

        if (num_segments >= MAX_SEGMENTS - 1) {
            fprintf(stderr, "Error: too many scheduling segments\n");
            exit(1);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Statistics                                                          */
/* ------------------------------------------------------------------ */

static int waiting_time(const Task *t)
{
    return t->end - t->arrival - t->burst;
}

static void print_statistics(Algo algo, int quantum)
{
    double total_wait = 0.0;

    printf("\n");
    if (algo == ALGO_RR) {
        /* Round Robin: first show every CPU turn, then per-task summary. */
        printf("RR (Time quantum = %d):\n\n", quantum);
        printf("%-6s %-11s %-9s %-13s\n",
               "PID", "Start Time", "End Time", "Running Time");
        for (int i = 0; i < num_segments; i++)
            printf("%-6d %-11d %-9d %-13d\n", segments[i].pid,
                   segments[i].start, segments[i].end,
                   segments[i].end - segments[i].start);

        printf("\n%-6s %-13s %-13s %-9s %-12s\n",
               "PID", "Arrival Time", "Running Time", "End Time",
               "Waiting Time");
        for (int i = 0; i < num_tasks; i++) {
            const Task *t = &tasks[i];
            printf("%-6d %-13d %-13d %-9d %-12d\n", t->pid, t->arrival,
                   t->burst, t->end, waiting_time(t));
            total_wait += waiting_time(t);
        }
    } else {
        /* FCFS / SJF: one line per task, listed in execution order. */
        printf("%s:\n\n", algo == ALGO_FCFS ? "FCFS" : "SJF");
        printf("%-6s %-13s %-11s %-9s %-13s %-12s\n",
               "PID", "Arrival Time", "Start Time", "End Time",
               "Running Time", "Waiting Time");
        for (int i = 0; i < num_finished; i++) {
            const Task *t = &tasks[finish_order[i]];
            printf("%-6d %-13d %-11d %-9d %-13d %-12d\n", t->pid, t->arrival,
                   t->start, t->end, t->burst, waiting_time(t));
            total_wait += waiting_time(t);
        }
    }
    printf("\nAverage Waiting Time: %g\n", total_wait / num_tasks);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s input_file [FCFS|RR|SJF] [time_quantum]\n"
            "  time_quantum (positive integer) is required for RR and is\n"
            "  ignored by FCFS and SJF.\n", prog);
}

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4) {
        usage(argv[0]);
        return 1;
    }

    Algo algo;
    int quantum = 0;

    if      (strcasecmp(argv[2], "FCFS") == 0) algo = ALGO_FCFS;
    else if (strcasecmp(argv[2], "RR")   == 0) algo = ALGO_RR;
    else if (strcasecmp(argv[2], "SJF")  == 0) algo = ALGO_SJF;
    else {
        fprintf(stderr, "Error: unknown algorithm '%s'\n", argv[2]);
        usage(argv[0]);
        return 1;
    }

    if (algo == ALGO_RR) {
        char *endp;
        if (argc != 4) {
            fprintf(stderr, "Error: RR requires a time quantum\n");
            usage(argv[0]);
            return 1;
        }
        quantum = (int)strtol(argv[3], &endp, 10);
        if (*endp != '\0' || quantum <= 0) {
            fprintf(stderr, "Error: time quantum must be a positive integer\n");
            return 1;
        }
    }

    if (read_tasks(argv[1]) != 0)
        return 1;

    printf("Loaded %d task(s) from %s.  Algorithm: %s", num_tasks, argv[1],
           algo == ALGO_FCFS ? "FCFS" : algo == ALGO_RR ? "RR" : "SJF");
    if (algo == ALGO_RR) printf(" (quantum %d)", quantum);
    printf("\n\n");

    simulate(algo, quantum);
    print_statistics(algo, quantum);
    return 0;
}
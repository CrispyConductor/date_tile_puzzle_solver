import datetime
from solver import solve_for_date

d = datetime.date(2000, 1, 1)
while True:
    print(f'Date {d.month:02d}/{d.day:02d}:')
    s = solve_for_date(d.month, d.day)
    print(s.solution_str())
    d += datetime.timedelta(days=1)
    if d.year != 2000:
        break


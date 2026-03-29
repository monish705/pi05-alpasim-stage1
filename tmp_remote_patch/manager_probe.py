import cereal.messaging as messaging
sm = messaging.SubMaster(["selfdriveState", "onroadEvents", "managerState"])
for _ in range(30):
    sm.update(1000)
not_running = []
for p in sm["managerState"].processes:
    if (not p.running) and p.shouldBeRunning:
        not_running.append({"name": p.name, "exitCode": p.exitCode, "running": p.running, "should": p.shouldBeRunning})
events = [e.name for e in sm["onroadEvents"] if any([e.noEntry, e.softDisable, e.immediateDisable])]
print({"engageable": sm["selfdriveState"].engageable, "active": sm["selfdriveState"].active, "not_running": not_running, "events": events})

class ThreatDetector:
    def inspect(self,event):
        flags=[]
        if event.get("failed_auth_count",0)>=5: flags.append("authentication_burst")
        if event.get("privilege_change"): flags.append("privilege_change")
        if event.get("unknown_device"): flags.append("unknown_device")
        return {"suspicious":bool(flags),"signals":flags}

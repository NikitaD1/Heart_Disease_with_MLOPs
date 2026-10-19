{{- define "heart-api.name" -}}{{ .Release.Name }}{{- end -}}
{{- define "heart-api.labels" -}}
app: {{ include "heart-api.name" . }}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

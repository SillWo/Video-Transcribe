import { Route, Routes, useLocation } from 'react-router-dom'

import { AppShell } from './components/AppShell'
import { ModelManagementProvider } from './context/ModelManagementContext'
import { useTranscriberController } from './hooks/useTranscriberController'
import { ModelsPage } from './pages/ModelsPage'
import { TranscriberPage } from './pages/TranscriberPage'

export default function App() {
  const location = useLocation()
  const transcriberController = useTranscriberController(location.pathname === '/')

  return (
    <ModelManagementProvider>
      <Routes>
        <Route
          path="/"
          element={
            <AppShell>
              <TranscriberPage controller={transcriberController} />
            </AppShell>
          }
        />
        <Route
          path="/models"
          element={
            <AppShell>
              <ModelsPage />
            </AppShell>
          }
        />
      </Routes>
    </ModelManagementProvider>
  )
}

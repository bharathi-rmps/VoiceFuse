import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:hexcolor/hexcolor.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:voicefuse/auth/sign_in.dart';
import 'package:voicefuse/firebase_options.dart';
import 'package:voicefuse/home_page.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);

  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late bool isLoggedIn;
  getLoginInfo() async {
    final prefs = await SharedPreferences.getInstance();
    final showHome = prefs.getBool('isLoggedIn') ?? false;
    setState(() {
      isLoggedIn = showHome;
    });
  }

  @override
  void initState() {
    super.initState();
    getLoginInfo();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: HexColor('E3DFFD')),
      home: !isLoggedIn ? const LoginPage() : const HomePage(),
    );
  }
}
